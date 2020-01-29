# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:47:29 2019

@author: fkurtog
"""

import numpy as np
import cv2
from dipy.segment.clustering import QuickBundles
import matplotlib.pyplot as plt
from tesseract import image_to_string
import sys


#%% Importing Image
img = cv2.imread('4.jpg')
plt.figure(0)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


#%% Cropping the image to eliminate 
crop_img = img[213:3030,0:img.shape[1],:]
bottom_img = img[3000:img.shape[0],2500:img.shape[1],:]
if __name__ == '__main__':
 
  if len(sys.argv) < 2:
    print('Usage: python ocr_simple.py image.jpg')
    sys.exit(1)
plt.imshow(cv2.cvtColor(bottom_img, cv2.COLOR_BGR2RGB))
config = ('-l eng --oem 1 --psm 3')
text = pytesseract.image_to_string(bottom_img, config=config)
print(text)

#%% Closing
kernel = np.ones((12,12),np.uint8)
closed_img = cv2.morphologyEx(crop_img, cv2.MORPH_CLOSE,kernel)
plt.figure(0)
##plt.imshow(cv2.cvtColor(closed_img, cv2.COLOR_BGR2RGB))

closed_img2 = cv2.morphologyEx(closed_img, cv2.MORPH_CLOSE,kernel)
plt.figure(1)
##plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#%% Filtering
blurred_img = cv2.medianBlur(closed_img, 5)
#plt.imshow(cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB))


#%%
imgray = cv2.cvtColor(closed_img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,230,255,0)
plt.figure(2)
plt.imshow(thresh,cmap='gray')

#%%
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

maxarea = 1000
for contour in contours:
    area = cv2.contourArea(contour)
    if (area > maxarea) and (area <8445184):
        BiggestContour = contour
        maxarea = area
plt.figure(3)         
cv2.drawContours(closed_img, BiggestContour,-1, (0,255,0), 10)
plt.imshow(cv2.cvtColor(closed_img, cv2.COLOR_BGR2RGB))



print('Tumor Radius =',np.sqrt(maxarea/np.pi),'micrometers')