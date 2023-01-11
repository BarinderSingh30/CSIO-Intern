import cv2 

#sift
sift = cv2.SIFT_create()

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


imgL = cv2.imread('teddy-png-2/teddy/imgL.png')  
imgR = cv2.imread('teddy-png-2/teddy/imgR.png') 

imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

keypoints_L, descriptors_L = sift.detectAndCompute(imgL,None)
keypoints_R, descriptors_R = sift.detectAndCompute(imgR,None)

matches = bf.match(descriptors_L,descriptors_R)
matches = sorted(matches, key = lambda x:x.distance)
print(len(matches))

img3 = cv2.drawMatches(imgL, keypoints_L, imgR, keypoints_R, matches[205:215], imgR, flags=2)

cv2.imshow('SIFT', img3)

cv2.waitKey(0)