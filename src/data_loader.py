import cv2
import os


folder_path = r"C:\Users\fhdud\PycharmProjects\pythonProject_Disser\data\images"
out_path = r"C:\Users\fhdud\PycharmProjects\pythonProject_Disser\data\gauss_masks"
images = os.listdir(folder_path)

for img in images:
    image_path = os.path.join(folder_path, img)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 164, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150)

    out_image_path = os.path.join(out_path, img)
    cv2.imwrite(out_image_path, edges)
