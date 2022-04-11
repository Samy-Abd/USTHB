import cv2
import numpy as np

img_ = cv2.imread('img3.jpg')
img_ = cv2.resize(img_, (0,0), fx=0.3, fy=0.3)
imgR = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)

img = cv2.imread('img1.jpg')
img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
imgL = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
#sift = cv.xfeatures2d.SIFT_create()
# find the key points and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imgL,None)
kp2, des2 = sift.detectAndCompute(imgR,None)

cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_,kp1,None))
cv2.waitKey(0)

#FLANN_INDEX_KDTREE = 0
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks = 50)
#match = cv2.FlannBasedMatcher(index_params, search_params)
#matches = match.knnMatch(des1,des2,k=2)

match = cv2.BFMatcher()
matches = match.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.3*n.distance:
        good.append(m)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)

img3 = cv2.drawMatches(img_,kp1,img,kp2,good,None,**draw_params)
cv2.imshow("original_image_drawMatches.jpg", img3)
cv2.waitKey(0)
cv2.imwrite("original_image_drawMatches.jpg", img3)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    h,w = imgL.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    imgR = cv2.polylines(imgR,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    cv2.imshow("original_image_overlapping.jpg", imgR)
else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))

#warped_image = cv2.warpPerspective(image, homography_matrix, dimension_of_warped_image)

dst = cv2.warpPerspective(img_,M,(img.shape[1] + img_.shape[1], img.shape[0]))
dst[0:img.shape[0], 0:img.shape[1]] = img
cv2.imshow("original_image_stitched.jpg", dst)
cv2.waitKey(0)

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-1])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:]) 
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-1])    
    return frame

cv2.imshow("original_image_stiched_crop.jpg", trim(dst))
cv2.waitKey(0)
cv2.imwrite("original_image_stiched_crop.jpg", trim(dst))
