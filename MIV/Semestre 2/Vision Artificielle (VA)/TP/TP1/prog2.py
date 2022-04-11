import cv2
import numpy as np

kernel_size = 3
scale = 1
delta = 0
ddepth = cv2.CV_16S
sigma=24

img = cv2.imread('im1.jpg')
img = cv2.GaussianBlur(img,(3,3),sigma,0, cv2.BORDER_DEFAULT)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray_lap = cv2.Laplacian(gray,ddepth,ksize = kernel_size,scale = scale,delta = delta)
dst = cv2.convertScaleAbs(gray_lap)

cv2.imshow('laplacian',dst)
cv2.imwrite("LAP_imag.png", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()