import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import os


def image_load(path: str, verbose: bool = True):
    """
    Загружает изображение в градациях серого (1 канал).

    Параметры:
        path: Путь к файлу изображения.

    Возвращает:
        numpy.ndarray (H, W) с типами uint8 [0, 255] или None при ошибке.
    """
    if not Path(path).exists():
        if verbose:
            print(f"Ошибка: файл не найден - '{path}'")
        raise FileNotFoundError(f"Image file not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        if verbose:
            print(f"Ошибка загрузки (возможно повреждённый файл): '{path}'")
        return None

    if verbose:
        print(f"Успешно загружено: {img.shape} | Диапазон: [{img.min()}, {img.max()}]")

    return img


def generate_markers_metall(gray_img):
    img_blur = cv2.bilateralFilter(gray_img, 9, 75, 75)

    # plt.imshow(img_blur, cmap='gray')
    # plt.title("Blur")
    # plt.colorbar()
    # plt.show()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_blur)

    # plt.imshow(img_eq, cmap='gray')
    # plt.title("Clahe")
    # plt.colorbar()
    # plt.show()

    _, binary = cv2.threshold(img_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = np.mean(binary == 255)
    if white_ratio > 0.8:
        binary = 255 - binary

    kernel = np.ones((4, 4), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transforms = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # plt.imshow(dist_transforms, cmap='gray')
    # plt.title("Distance Transform")
    # plt.colorbar()
    # plt.show()

    _, sure_fg = cv2.threshold(dist_transforms, 0.2 * dist_transforms.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    markers = cv2.connectedComponents(sure_fg)[1]
    markers = markers + 1  # конвертируем цвета
    markers[unknown == 255] = 0

    return markers


def watershed_img(img, markers):
    assert len(img.shape) == 2
    assert img.dtype == np.uint8
    assert markers.dtype == np.int32

    img_rgb = cv2.merge([img] * 3)
    markers_ws = cv2.watershed(img_rgb, markers.copy())

    img_with_boundaries = img_rgb.copy()
    img_with_boundaries[markers_ws == -1] = [255, 0, 0]

    segments_colored = np.zeros_like(img_rgb)
    for label in np.unique(markers_ws):
        if label <= 1:
            continue
        mask = markers_ws == label
        color = np.random.randint(0, 255, 3)
        segments_colored[mask] = color

    return img_with_boundaries, segments_colored, markers_ws


if __name__ == "__main__":
    input_folder = r"C:\Users\fhdud\PycharmProjects\pythonProject_Disser\data\images"
    output_folder = r"C:\Users\fhdud\PycharmProjects\pythonProject_Disser\data\markers"
    orig_images = os.listdir(input_folder)

    for orig_img in orig_images:
        image_path = os.path.join(input_folder, orig_img)

        image_loads = image_load(image_path)
        markers = generate_markers_metall(image_loads)
        img_with_boundaries, segments_colored, markers_ws = watershed_img(image_loads, markers)

        segments_colored_bgr = cv2.cvtColor(segments_colored, cv2.COLOR_RGB2BGR)

        base_filename = os.path.splitext(orig_img)[0]
        out_image_path = os.path.join(output_folder, base_filename + ".png")

        cv2.imwrite(out_image_path, segments_colored_bgr)



    # plt.imshow(image_loads, cmap='gray')
    # plt.title("Original image")
    # plt.axis('off')
    # plt.show()

    # plt.imshow(markers, cmap='gray')
    # plt.title("Markers for Watershed")
    # plt.axis('off')
    # plt.show()

    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img_with_boundaries)
    # plt.title("Границы сегментации")
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(segments_colored)
    # plt.title("Цветная сегментация")
    # plt.axis('off')
    # plt.show()