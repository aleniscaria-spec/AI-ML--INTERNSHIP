import cv2
import numpy as np

img = cv2.imread("Image_classification/image.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX)
magnitude = magnitude.astype(np.uint8)

cv2.imwrite("edges.png", magnitude)
cv2.imwrite("gray.png", gray)
cv2.imwrite("sobel_x.png", cv2.convertScaleAbs(sobel_x))
cv2.imwrite("sobel_y.png", cv2.convertScaleAbs(sobel_y))
cv2.imwrite("edges.png", magnitude)
cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)
cv2.imshow("Sobel X",cv2.convertScaleAbs(sobel_x))
cv2.imshow("Sobel Y", cv2.convertScaleAbs(sobel_y))
cv2.imshow("Final Edge Detection",magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()