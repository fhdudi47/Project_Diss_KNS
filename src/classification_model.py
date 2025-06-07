import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import functional as TF


class CustomImageDataset(data.Dataset):
    def __init__(self, data_dir, image_size=(512, 512)):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Lambda(lambda img: TF.rotate(img, angle=random.choice([90, 180, 270]))),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.dataset = datasets.ImageFolder(data_dir, transform=self.transform)
        self.class_names = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def split_dataset_stratified(dataset, val_split=0.2, test_split=0.1):
    targets = [sample[1] for sample in dataset]
    indices = list(range(len(dataset)))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_split, stratify=targets, random_state=42)

    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_split / (1 - test_split),
        stratify=[targets[i] for i in train_val_idx], random_state=42)

    return train_idx, val_idx, test_idx


def get_loaders(data_dir, image_size=(512, 512), batch_size=32, val_split=0.2, test_split=0.1, shuffle=True):
    full_dataset = CustomImageDataset(data_dir, image_size)
    train_idx, val_idx, test_idx = split_dataset_stratified(full_dataset, val_split, test_split)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_dataset.class_names


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=3): # 3 classes bimodal, globular, lamellar
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * BasicBlock.expansion

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, save_path='best_model.pth'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
        self.device = device
        self.save_path = save_path
        self.best_val_acc = 0.0

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1 = []

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        return train_loss, train_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / total
        val_acc = correct / total
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return val_loss, val_acc, precision, recall, f1

    def evaluate_on_loader(self, data_loader, label="Dataset"):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / total
        accuracy = correct / total
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"\n{label} evaluation:")
        print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        return avg_loss, accuracy, precision, recall, f1

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(18, 10))

        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Val Loss')
        plt.legend()
        plt.title("Loss")

        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Train Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Val Accuracy')
        plt.legend()
        plt.title("Accuracy")

        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.val_precisions, label='Precision')
        plt.plot(epochs, self.val_recalls, label='Recall')
        plt.plot(epochs, self.val_f1, label='F1')
        plt.legend()
        plt.title("Precision / Recall / F1")

        plt.tight_layout()
        plt.show()

    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, precision, recall, f1 = self.validate()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_precisions.append(precision)
            self.val_recalls.append(recall)
            self.val_f1.append(f1)

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
            print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")
            print(f"Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            self.scheduler.step(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model()


# !!!!!!!!!!!!!!!!!!!!
def load_model(model_class, model_path, num_classes, device):
    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(image_path, model, class_names, image_size=(512, 512), device='cpu'):
    if not isinstance(device, str):
        device = str(device)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy().flatten()
        predicted_class = class_names[probs.argmax()]
        print(f"\n🖼️ Prediction: {predicted_class}")
        for cls, prob in zip(class_names, probs):
            print(f"{cls}: {prob:.4f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data', 'image_class')

    # Параметры
    image_size = (512, 512)
    batch_size = 32
    epochs = 50
    learning_rate = 0.001

    # Загрузка данных
    train_loader, val_loader, test_loader, class_names = get_loaders(data_dir, image_size=image_size, batch_size=batch_size)
    num_classes = len(class_names)

    # Инициализация модели, loss, optimizer
    model = ResNet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)

    # Обучение
    trainer.fit(epochs)

    # Оценка на тесте
    trainer.evaluate_on_loader(test_loader, label="Test")

    # Визуализация метрик
    trainer.plot_metrics()

    test_image_path = r"D:\fhdud\Documents\Изображения\redkie\redkie_012.jpg"
    print("\nRunning inference on:", test_image_path)
    loaded_model = load_model(ResNet18, 'best_model.pth', num_classes=num_classes, device=device)
    predict_image(test_image_path, loaded_model, class_names, device=device)












