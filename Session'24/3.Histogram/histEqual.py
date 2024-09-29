# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:59:10 2022

@author: tamim
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt

#%%

img = cv2.imread("histogram.jpg", cv2.IMREAD_GRAYSCALE)
plt.figure(6)
plt.title("Input Image")
plt.imshow(img,"gray")
plt.show()
plt.figure(7)
plt.title("Input Image Histogram")
plt.hist(img.ravel(),256,[0,255])
plt.show()

#%%

a = np.zeros(256, dtype='float64')
p = np.zeros(256, dtype='float64')
s = np.zeros(256, dtype='float64')
c = np.zeros(256, dtype='float64')
a_o = np.zeros(256, dtype='float64')
p_o = np.zeros(256, dtype='float64')
s_o = np.zeros(256, dtype='float64')
c_o = np.zeros(256, dtype='float64')
h = img.shape[0]
w = img.shape[1]

for i in range(h):
    for j in range(w):
        x = img[i][j]
        a[x] = a[x]+1

ln = h*w

count = np.zeros(256, dtype='float64')
for i in range (256):
    count[i] = i
    p[i] = a[i]/ln
    
plt.figure(1)
plt.plot(count[:], p, color="red", label="PDF") 
plt.show()    
sum =0 
for i in range(256):
        sum = sum + p[i]
        c[i] = sum
        s[i] = round(sum*255)
print(c)
plt.figure(2)
plt.plot(count[:], c, label="CDF")
plt.show()    
for i in range(h):
    for j in range(w):
        
        k = img[i][j]
        img[i][j] = s[k]

#%%

for i in range(h):
    for j in range(w):
        x = img[i][j]
        a_o[x] = a_o[x]+1
        
for i in range (256):
    p_o[i] = a_o[i]/ln
plt.figure(10)
plt.plot(count[:], p_o, color="red", label="PDF_output") 
plt.show()  


sum =0 
for i in range(256):
        sum = sum + p_o[i]
        c_o[i] = sum
        s_o[i] = round(sum*255)
print(c_o)
a = [x for x in range(256)] 
print(a)
plt.figure(12)
plt.plot(a[:], c_o, label="CDF_output")
plt.show()   


plt.figure(4)
plt.title("Output image")
plt.imshow(img,"gray")
plt.show()
plt.figure(3)
plt.title("Output Image Histogram")
plt.hist(img.ravel(),256,[0,255])
plt.show()