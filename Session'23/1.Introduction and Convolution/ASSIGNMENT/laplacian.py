# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('moon.tif',cv2.IMREAD_GRAYSCALE)
cv2.imshow('input', img)
k = (1.0/4)*np.array(
        [[0,1,0],
         [1,-4,1],
         [0,1,0]]
        )
lap = cv2.filter2D(img,-1, k)

r, c = img.shape
new_image = np.zeros((r, c))
L_sunnzi = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])          
for i in range(r-2):
    for j in range(c-2):
        new_image[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * L_sunnzi))
new_image=np.uint8(new_image)
cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
img = np.round(img).astype(np.uint8)    
#cv2.imshow('outputLaplacefor3', img)  
plt.title("Laplacian Operator")
plt.imshow(new_image, cmap="gray", vmin=0, vmax=255)
plt.set_cmap('gray')
plt.imshow(new_image)


img = cv2.imread('cup.jpg',cv2.IMREAD_GRAYSCALE)
dest = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
dest1 = cv2.convertScaleAbs(dest)

cv2.normalize(dest, dest, 0, 255, cv2.NORM_MINMAX)
dest2 = np.round(dest).astype(np.uint8)
'''
plt.imshow(dest1, cmap="gray")
plt.imshow(dest1)
plt.imshow(dest2, cmap="gray")
'''
laplace = (1.0/16) * np.array(
        [[0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0]])     
    
for i in range(5, img.shape[0]-5):
    for j in range(5, img.shape[1]-5):
        sum = 0.0
        for x in range(5):
            for y in range(5):
                a = img.item(i+x, j+y)
                w = laplace[x][y]
                sum = sum + (w * a)
        b = sum
        img.itemset((i,j), b)
#cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
#img = np.round(img).astype(np.uint8)

img = cv2.convertScaleAbs(img)
cv2.imshow('outputLaplacefor5', img)  

cv2.waitKey(0)
cv2.destroyAllWindows()
