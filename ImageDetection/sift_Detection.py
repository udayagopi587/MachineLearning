# -*- coding: utf-8 -*-
"""
Name:- Udaya Gopi K
Conc:- SIFT
"""

import cv2 as cv

#sift
sift = cv.SIFT_create() #Detector of similarity


#ORB is comparitively faster for feature detector.

#feature matching
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)



img1 = cv.imread('road.jpg') #If we add flag as 0 then it will automatically convert to grascale image
scale_percent = 60 # percent of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized1 = cv.resize(img1, dim, interpolation = cv.INTER_AREA)  
cv.imshow("Road I/p", resized1)

img2 = cv.imread('roadcrop.jpg') 

scale_percent = 60 # percent of original size
width = int(img2.shape[1] * scale_percent / 100)
height = int(img2.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized2 = cv.resize(img2, dim, interpolation = cv.INTER_AREA)  
cv.imshow("RoadCrop", resized2)

img1 = cv.cvtColor(resized1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(resized2, cv.COLOR_BGR2GRAY)

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)


##### Looking into Descriptors ######

print(descriptors_1) #This gives a 2D array with features, descriptors, ex:- (2942, 128) which means it has 2942 features and each features with 128 descriptors.


#Drawing the keypoints..

# =============================================================================
# kpimg1 = cv.drawKeypoints(img1, keypoints_1, None)
# kpimg2 = cv.drawKeypoints(img2, keypoints_2, None)
# 
# cv.imshow('KeyPoints1', kpimg1)
# cv.imshow('KeyPoints2', kpimg2)
# =============================================================================

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance) #Sorting based on the most likely match to the least likely match.

######### Looking for no.of good matches ##########
##Link:- https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_feature2d/py_matcher/py_matcher.html

# =============================================================================
# good = []
# 
# for m,n in matches:
#     if m.distance > 0.75*n.distance:
#         good.append([m])
# 
# print(len(good))
# =============================================================================

img3 = cv.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:600], img2, flags=2)

cv.imshow('SIFT', img3)

cv.waitKey(0)