from PIL import Image
import os


folder_path = r"C:\Users\fhdud\OneDrive\Рабочий стол\Диссертация и все что связано с ней\Фото Микроструктуры"
out_path = r"/data/images"
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

masks_folder_path = r"C:\Users\fhdud\PycharmProjects\pythonProject_Disser\data\masks"
out_masks_folder_path = r"C:\Users\fhdud\PycharmProjects\pythonProject_Disser\data\images"
files = os.listdir(masks_folder_path)

for idx, file_name in enumerate(files):
    new_name = f"img{idx}{os.path.splitext(file_name)[1]}"

    old_file_path = os.path.join(masks_folder_path, file_name)
    new_file_path = os.path.join(out_masks_folder_path, new_name)

    os.rename(old_file_path, new_file_path)


if __name__ == "__main__":
    pass
