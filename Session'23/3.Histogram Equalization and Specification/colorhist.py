# -*- coding: utf-8 -*-
"""
Created on Mon May  1 18:41:27 2023

@author: USER
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(r"H:\\Other computers\\My Laptop\\1.1-4.2STUDY\\COURSE TAKEN\2023\\Image Lab\\labtest-18\\histogram_.jpg")

plt.figure(figsize=(20, 10))



color = ('b','g','r')
 
plt.subplot(2, 2, 1)
for i,col in enumerate(color):
     
    #To use OpenCV's calcHist function, uncomment below
    #histr = cv2.calcHist([colorimage],[i],None,[256],[0,256])
     
    #To use numpy histogram function, uncomment below
    histr, _ = np.histogram(img[:,:,i],256,[0,256])
    plt.plot(histr,color = col)  #Add histogram to our plot 
    #plt.xlim([0,256])
     
plt.show() 

# For ease of understanding, we explicitly equalize each channel individually
colorimage_b = cv2.equalizeHist(img[:,:,0])
colorimage_g = cv2.equalizeHist(img[:,:,1])
colorimage_r = cv2.equalizeHist(img[:,:,2])
 
# Next we stack our equalized channels back into a single image
colorimage_e = np.stack((colorimage_b,colorimage_g,colorimage_r), axis=2)
colorimage_e.shape
# Using Numpy to calculate the histogram

plt.subplot(2, 2, 2)
color = ('b','g','r')
for i,col in enumerate(color):
    histr, _ = np.histogram(colorimage_e[:,:,i],256,[0,256])
    plt.plot(histr,color = col)
    #plt.xlim([0,256])


plt.show()

cv2.imshow("input",img)
cv2.imshow("output",colorimage_e)

# Convert image from RGB to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

plt.subplot(2, 2, 3)
histr, _ = np.histogram(img_hsv[:,:,2],256,[0,256])
plt.plot(histr,color = 'b')

# Histogram equalisation on the V-channel
img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

plt.subplot(2, 2, 4)
histr, _ = np.histogram(img_hsv[:,:,2],256,[0,256])
plt.plot(histr,color = 'g')


# Convert image back from HSV to RGB
image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


'''
color = ('b','g','r')
for i,col in enumerate(color):
    histr, _ = np.histogram(image[:,:,i],256,[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])'''
plt.show()



cv2.imshow("output2",image)
cv2.waitKey(0)
cv2.destroyAllWindows()