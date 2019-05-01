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
plt.figure(0)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


#%% Cropping the image to eliminate 
crop_img = img[213:3030,0:img.shape[1],:]
plt.figure(1)
plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

#%% Closing
kernel = np.ones((12,12),np.uint8)
closed_img = cv2.morphologyEx(crop_img, cv2.MORPH_CLOSE,kernel)
plt.figure(2)
plt.imshow(cv2.cvtColor(closed_img, cv2.COLOR_BGR2RGB))

#%% Filtering
#blurred_img = cv2.medianBlur(closed_img, 5)
#plt.imshow(cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB))


#%% 
closed_img = np.float32(closed_img)
#%%

vector_feature = IdentityFeature()
metric = EuclideanMetric(feature=vector_feature)
qb = QuickBundles(threshold=120000, metric=metric)
clusters = qb.cluster(closed_img)

print("Nb. clusters:", len(clusters))


imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#indices=clusters[1]
#A=closed_img[indices,:,:]


#%%
for j, group in enumerate(clusters):
    if len(group)>1:
        for indx in group.indices:
                
    #            print(indx)
            cm = closed_img[indx]
            cm = (cm[0],cm[1])
            color_cm_list.append((cm, 255.999*colors[indices_shuf_list[j]]))
            _, cluster_color,_,_ = cv2.floodFill(cluster_color,None,cm,255.999*colors[indices_shuf_list[j]])
    #            a = 255.999*colors[indices_shuf_list[j]]
            _, color_by_index, _, _ = cv2.floodFill(color_by_index,None,cm,int(indices_shuf_list[j]+1))
            i+=1
        else:
            for indx in group.indices:
                cm = closed_img[indx]
                cm = (cm[0],cm[1])
                color_cm_list.append((cm, (255,255,255)))
                _, cluster_color,_,_ = cv2.floodFill(cluster_color,None,cm,(255,255,255))
                _, color_by_index, _, _ = cv2.floodFill(color_by_index,None,cm,2+len(clusters))
      
#    figure()
#    imshow(cluster_color)
    cluster_color[...,0] *= binary_data
    cluster_color[...,1] *= binary_data
    cluster_color[...,2] *= binary_data
    
    
#    mask = cluster_color == np.array([255, 255, 0])
#    figure()
    #imshow(mask.astype('f4'), cmap='gray')
#    bmask = mask.sum(axis=-1)
#    cluster_color[np.where(bmask == 3 )] = (255,0,255)
    
#    figure()
#    imshow(cluster_color)
    imwrite(save_name,cluster_color)
