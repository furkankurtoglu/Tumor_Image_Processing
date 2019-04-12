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

#%% Closing
kernel = np.ones((12,12),np.uint8)
closed_img = cv2.morphologyEx(crop_img, cv2.MORPH_CLOSE,kernel)
plt.figure(0)
plt.imshow(cv2.cvtColor(closed_img, cv2.COLOR_BGR2RGB))

closed_img2 = cv2.morphologyEx(closed_img, cv2.MORPH_CLOSE,kernel)
plt.figure(1)
plt.imshow(cv2.cvtColor(closed_img2, cv2.COLOR_BGR2RGB))

#%% Filtering
blurred_img = cv2.medianBlur(closed_img, 5)
#plt.imshow(cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB))


#%%
imgray = cv2.cvtColor(blurred_img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,160,255,0)
plt.imshow(thresh,cmap='gray')

#%%
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

maxarea = 1000
for contour in contours:
    area = cv2.contourArea(contour)
    if (area > maxarea) and (area <8445184):
        BiggestContour = contour
        print(area)
        maxarea = area
        
cv2.drawContours(closed_img, BiggestContour,-1, (0,255,0), 10)
plt.imshow(cv2.cvtColor(closed_img, cv2.COLOR_BGR2RGB))