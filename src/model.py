import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms.v2 as tfs_v2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class SegmentDataset(Dataset):
    def __init__(self, root_dir, subset: str, transform_img=None, transform_mask=None):
        assert subset in ("train", "val", "test")
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        img_dir = os.path.join(root_dir, "segmentations")
        mask_dir = os.path.join(root_dir, "binary_masks")

        # Словари: {имя_без_расширения: путь_до_файла}
        image_files = {
            os.path.splitext(f)[0]: os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith(('.png', '.jpg'))
        }
        mask_files = {
            os.path.splitext(f)[0]: os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.endswith(('.png', '.jpg'))
        }

        # Только общие ключи
        common_keys = sorted(list(set(image_files.keys()) & set(mask_files.keys())))
        self.images = [image_files[k] for k in common_keys]
        self.masks = [mask_files[k] for k in common_keys]

        # Разделение по subset
        n = len(self.images)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        if subset == "train":
            self.images = self.images[:n_train]
            self.masks = self.masks[:n_train]
        elif subset == "val":
            self.images = self.images[n_train:n_train + n_val]
            self.masks = self.masks[n_train:n_train + n_val]
        else:
            self.images = self.images[n_train + n_val:]
            self.masks = self.masks[n_train + n_val:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask = (mask > 0.5).float()

        return img, mask


class UNetModel(nn.Module):
    class _TwoConvLayers(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )

        def forward(self, x):
            return self.model(x)

    class _EncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)
            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.block(x)
            y = self.max_pool(x)
            return y, x

    class _DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)

        def forward(self, x, y):
            x = self.transpose(x)
            u = torch.cat([x, y], dim=1)
            return self.block(u)

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc1 = self._EncoderBlock(in_channels, 64)
        self.enc2 = self._EncoderBlock(64, 128)
        self.enc3 = self._EncoderBlock(128, 256)
        self.enc4 = self._EncoderBlock(256, 512)

        self.bottleneck = self._TwoConvLayers(512, 1024)

        self.dec1 = self._DecoderBlock(1024, 512)
        self.dec2 = self._DecoderBlock(512, 256)
        self.dec3 = self._DecoderBlock(256, 128)
        self.dec4 = self._DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x, y1 = self.enc1(x)
        x, y2 = self.enc2(x)
        x, y3 = self.enc3(x)
        x, y4 = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec1(x, y4)
        x = self.dec2(x, y3)
        x = self.dec3(x, y2)
        x = self.dec4(x, y1)

        return self.out(x)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = targets.size(0)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        inter = (m1 * m2).sum(1)
        dice = (2 * inter + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        return 1 - dice.mean()

def compute_metrics(pred_logits, true_masks, threshold=0.5):
    probs = torch.sigmoid(pred_logits).cpu().numpy().reshape(-1)
    preds = (probs >= threshold).astype(np.uint8)
    trues = true_masks.cpu().numpy().reshape(-1).astype(np.uint8)
    return {
        "acc": accuracy_score(trues, preds),
        "prec": precision_score(trues, preds, zero_division=0),
        "rec": recall_score(trues, preds, zero_division=0),
        "f1": f1_score(trues, preds, zero_division=0),
    }


def plot_metrics(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label='Train Loss')
    plt.plot(epochs, history["val_loss"], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label='Accuracy')
    plt.plot(epochs, history["val_prec"], label='Precision')
    plt.plot(epochs, history["val_rec"], label='Recall')
    plt.plot(epochs, history["val_f1"], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cpu")

    transform_img = tfs_v2.Compose([
        tfs_v2.ToImage(),
        tfs_v2.Resize((256, 256)),
        tfs_v2.ToDtype(torch.float32, scale=True),
        tfs_v2.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    transform_mask = tfs_v2.Compose([
        tfs_v2.ToImage(),
        tfs_v2.Resize((256, 256)),
        tfs_v2.ToDtype(torch.float32, scale=True)
    ])

    ds_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "image_class"))
    train_set = SegmentDataset(ds_root, "train", transform_img, transform_mask)
    val_set = SegmentDataset(ds_root, "val", transform_img, transform_mask)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    model = UNetModel().to(device)
    loss_fn1 = nn.BCEWithLogitsLoss()
    loss_fn2 = SoftDiceLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    epochs = 50
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_prec": [], "val_rec": [], "val_f1": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn1(logits, y) + loss_fn2(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        history["train_loss"].append(train_loss / len(train_loader))

        model.eval()
        val_loss, all_preds, all_labels = 0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                val_loss += (loss_fn1(logits, y) + loss_fn2(logits, y)).item()
                all_preds.append(logits)
                all_labels.append(y)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_preds, all_labels)

        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(metrics["acc"])
        history["val_prec"].append(metrics["prec"])
        history["val_rec"].append(metrics["rec"])
        history["val_f1"].append(metrics["f1"])

        print(f"Epoch {epoch+1}: {metrics}")

    torch.save(model.state_dict(), "Model_UNet_CPU.pth")
    plot_metrics(history)
