# import computer vision library(cv2) in this code
import cv2

img_path = "im1.jpg"
image_c = cv2.imread(img_path)
image = cv2.cvtColor(image_c,cv2.COLOR_BGR2GRAY)
#image=image_c

# mentioning absolute path of the image
#cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)
#borderType: This specify boundaries of an image while kernel is applied on borders of an image.
# cv2.BORDER_DEFAULT: gfedcb|abcdefgh|gfedcba

blur_img = cv2.GaussianBlur(image,(3,3),6.0,0, cv2.BORDER_DEFAULT)

    # show the image on the newly created image window
	#cv2.imshow('Blur image',blur_img)
cv2.imwrite("blur_imag.png", blur_img)
cv2.waitKey(0)
