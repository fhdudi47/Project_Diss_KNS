from PIL import Image
import os
import numpy as np


# Путь к папке с изображениями
folder_path = r'C:\Users\fhdud\OneDrive\Рабочий стол\Диссертация и все что связано с ней\Фото Микроструктуры'

# Получаем список файлов в папке
files = os.listdir(folder_path)

# Проходимся по циклу для обработки изображений
for image_name in files:
    image_path = os.path.join(folder_path, image_name)

    # Открываем изображение с помощью PIL
    image = Image.open(image_path)

    # Проверяем размер изображения
    if image.size == (472, 377):  # В PIL size это (width, height)
        continue
    else:
        print(image.size)

    # Преобразуем изображение в оттенки серого, если оно не в сером
    if image.mode != 'L':  # Если изображение не в серых то преобразуем
        gray_image = image.convert('L')
    else:
        gray_image = image

    # Преобразуем изображение в массив numpy
    gray_image = np.array(gray_image)

    # Преобразуем матрицу в вектор
    vector_repr = gray_image.flatten()

    # Нормируем значения в пределах от 0 до 1
    vector_repr = vector_repr / 255.0

    print(vector_repr.shape)

