# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:47:29 2019

@author: fkurtog
"""

import numpy as np
import cv2
from dipy.segment.clustering import QuickBundles
import matplotlib.pyplot as plt



#%% Importing Image
img = cv2.imread('final.jpg')
#plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


#%% Cropping the image to eliminate 
crop_img = img[213:3030,0:img.shape[1]]
#plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))


#%% Filtering
blurred_frame = cv2.medianBlur(crop_img, 5)
plt.imshow(cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB))


#%%
 
imgray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,30,255,0)
#plt.imshow(thresh)


im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

maxarea = 1
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 10:
        BiggestContour = contour
        print(area)
        z = cv2.drawContours(crop_img, BiggestContour, 0, (0,255,0), 100)
        plt.imshow(cv2.cvtColor(z, cv2.COLOR_BGR2RGB))