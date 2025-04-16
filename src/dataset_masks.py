import cv2
import os

path = r"C:\Users\fhdud\PycharmProjects\pythonProject_Disser\data\images\img0.jpg"
image = os.path.join(path)

if not os.path.isfile(image):
    print(f"Не удается найти файл: {image}")
else:
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

blur0 = cv2.GaussianBlur(img, (5, 5), 0)
blur1 = cv2.GaussianBlur(img, (5, 5), 0)
blur2 = cv2.GaussianBlur(img, (5, 5), 0)
blur3 = cv2.GaussianBlur(img, (5, 5), 0)

edges00 = cv2.Canny(blur0, 250, 50)
edges10 = cv2.Canny(blur1, 50, 25)
edges20 = cv2.Canny(blur2, 150, 100)
edges30 = cv2.Canny(blur3, 200, 150)

edges0 = cv2.Canny(img, 250, 50)
edges1 = cv2.Canny(img, 50, 25)
edges2 = cv2.Canny(img, 150, 100)
edges3 = cv2.Canny(img, 200, 150)

cv2.imshow("im0", edges00)
cv2.imshow("im1", edges10)
cv2.imshow("im2", edges20)
cv2.imshow("im3", edges30)

cv2.imshow("im00", edges0)
cv2.imshow("im10", edges1)
cv2.imshow("im20", edges2)
cv2.imshow("im30", edges3)

cv2.waitKey(0)
cv2.destroyAllWindows()
