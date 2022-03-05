# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:47:29 2019

@author: fkurtog
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
from tesseract import image_to_string
import sys
=======
>>>>>>> Stashed changes


#%% Importing Image
img = cv2.imread('4.jpg') #You can change there are four images that are named "1,2,3,4"
plt.figure(0)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


#%% Cropping the image to eliminate 
crop_img = img[213:3030,0:img.shape[1],:]
<<<<<<< Updated upstream
bottom_img = img[3000:img.shape[0],2500:img.shape[1],:]
if __name__ == '__main__':
 
  if len(sys.argv) < 2:
    print('Usage: python ocr_simple.py image.jpg')
    sys.exit(1)
plt.imshow(cv2.cvtColor(bottom_img, cv2.COLOR_BGR2RGB))
config = ('-l eng --oem 1 --psm 3')
text = pytesseract.image_to_string(bottom_img, config=config)
print(text)
=======

>>>>>>> Stashed changes

#%% Closing
kernel = np.ones((12,12),np.uint8)
closed_img = cv2.morphologyEx(crop_img, cv2.MORPH_CLOSE,kernel)
plt.figure(1)
plt.title('Cropped Image')
plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))


#%% Filtering
blurred_img = cv2.medianBlur(closed_img, 15)
plt.figure(3)
plt.title('Blurred Image')
plt.imshow(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))


<<<<<<< Updated upstream
#%%
imgray = cv2.cvtColor(closed_img,cv2.COLOR_BGR2GRAY)
=======
#%% Thresholding
imgray = cv2.cvtColor(blurred_img,cv2.COLOR_BGR2GRAY)
>>>>>>> Stashed changes

ret,thresh = cv2.threshold(imgray,250,10,0)
plt.figure(4)
plt.title('Binary Labeled Image')
plt.imshow(thresh,cmap='gray')

#%% Finding and Drawing Contours
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

maxarea = 31400.0 #This value is the smallest tumor region
BiggestContours=[]
Areas=[]
for contour in contours:
    area = cv2.contourArea(contour)
    if (area > maxarea) and (area <8445184):  # This higher value is to not get a contour of whole image.
        BiggestContour = contour
        Areas.append(area)
        BiggestContours.append(contour)
        cv2.drawContours(closed_img, BiggestContour,-1, (0,255,0), 20)
        
plt.figure(5)
plt.title('Contour Plot')
plt.imshow(cv2.cvtColor(closed_img, cv2.COLOR_BGR2RGB))


#%% Calculating the Tumor and Necrotic Radius
if len(Areas) == 1:
    print('No Necrotic Area')
    Tumor_Radius = np.sqrt(max(Areas)/np.pi)
    print('Tumor Radius =',Tumor_Radius,'micrometers')
    
if len(Areas) == 2:
    print('There is a necrotic Area')
    Tumor_Radius = np.sqrt(max(Areas)/np.pi)
    Necrotic_Radius = np.sqrt(min(Areas)/np.pi)
    print('Tumor Radius =',Tumor_Radius,'micrometers')
    print('Necrotic Radius = ',Necrotic_Radius,'micrometers')