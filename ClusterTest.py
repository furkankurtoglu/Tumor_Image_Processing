# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:47:29 2019

@author: fkurtog
"""

import numpy as np
import cv2
from dipy.segment.clustering import QuickBundles
import matplotlib.pyplot as plt
from dipy.segment.metric import IdentityFeature, Feature
from dipy.segment.metric import AveragePointwiseEuclideanMetric, EuclideanMetric, SumPointwiseEuclideanMetric

#%%
import dipy.viz

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

#%% Filtering
#blurred_img = cv2.medianBlur(closed_img, 5)
#plt.imshow(cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB))


#%% 
closed_img = np.float32(closed_img)

vector_feature = IdentityFeature()
metric = EuclideanMetric(feature=vector_feature)
qb = QuickBundles(threshold=100000, metric=metric)
clusters = qb.cluster(closed_img)

print("Nb. clusters:", len(clusters))


imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

interactive = False
ren = window.Renderer()
ren.SetBackground(1, 1, 1)
window.record(ren,imgray)
if interactive:
    window.show(ren)