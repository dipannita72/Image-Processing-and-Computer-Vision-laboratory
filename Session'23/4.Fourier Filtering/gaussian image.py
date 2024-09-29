# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:04:07 2023

@author: D
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dpc
from math import sqrt,exp

def guassian_kernel(start, size, sigma):
    
    x, y = np.mgrid[start:size,start:size]
    normal = 1/(2*np.pi*(sigma**2))
    g = np.exp(-((x**2 + y**2)/(2*sigma**2))) * normal
    return g


img1 = guassian_kernel(0, 512, 180)
img2 = guassian_kernel(0, 512,100)
img1 = cv2.flip(img1,-1)

img3 = guassian_kernel(0, 256, 50)
img3 = cv2.flip(img3,-1)
img4 = guassian_kernel(0, 256,80)
img4 = cv2.flip(img4,0)
img5 = guassian_kernel(0, 256, 60)
img5 = cv2.flip(img5,1)
img6 = guassian_kernel(0, 256,10)

mask = img1 + img2 

cv2.normalize(img1,img1, 0, 255, cv2.NORM_MINMAX)
img1 = np.round(img1).astype(np.uint8)

cv2.normalize(img2,img2, 0, 255, cv2.NORM_MINMAX)
img2 = np.round(img2).astype(np.uint8)

mask2 = img2 #+ img2 
cv2.normalize(mask2,mask2, 0, 255, cv2.NORM_MINMAX)
mask2 = np.round(mask2).astype(np.uint8)

#cv2.imshow("I1",img1)
#cv2.imshow("I2",img2)
#cv2.imshow("final",mask)
cv2.imshow("final2",mask2)


cv2.imwrite('H:\\Other computers\\My Laptop\\1.1-4.2STUDY\\COURSE TAKEN\\2023\\Image Lab\\lab4\\mask.jpg', mask2)

img =  cv2.imread("lena.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("lkj",img )
final_img = cv2.add(img, mask2)
cv2.normalize(final_img,final_img, 0, 255, cv2.NORM_MINMAX)
final_img = np.round(final_img).astype(np.uint8)

cv2.imshow("gfgf",final_img)
cv2.imwrite('H:\\Other computers\\My Laptop\\1.1-4.2STUDY\\COURSE TAKEN\\2023\\Image Lab\\lab4\\cor_img.jpg',final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

