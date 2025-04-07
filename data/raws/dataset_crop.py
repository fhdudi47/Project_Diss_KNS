from PIL import Image
import os


folder_path = r"C:\Users\fhdud\OneDrive\Рабочий стол\Диссертация и все что связано с ней\Фото Микроструктуры"
out_path = r"C:\Users\fhdud\PycharmProjects\pythonProject_Disser\data\images"
images = os.listdir(folder_path)


for img in images:
    image_path = os.path.join(folder_path, img)

    try:
        image = Image.open(image_path)

        if image.size == (472, 377):
            image = image.crop((0, 0, 372, 372))
        else:
            print(f"{img}: original size {image.size}.")
            image = image.crop((0, 0, 372, 372))

        out_path_images = os.path.join(out_path, img)
        image.save(out_path_images)

    except Exception as e:
        print(f"Error during image processing: {img}.")





