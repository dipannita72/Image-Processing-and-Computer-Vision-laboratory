# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:44:11 2024

@author: USER
"""

import cv2
import numpy as np
import scipy as sp
import scipy.signal as sg
import matplotlib.pyplot as plt
   



def convolve(img,k):
    w = img.shape[0]
    h = img.shape[1]
    print(w,h)
    out = np.zeros(shape=(w, h))
    p = k.shape[0] // 2+1
    for i in range(5,img.shape[0]-5):
        for j in range(5,img.shape[1]-5):
            sum=0.0
            for x in range(5):
                for y in range(5):
                    a= img.item(i+x,j+y)
                    b=k[x][y]*a
                    sum=sum+b
                    #print(sum)
            out.itemset((i,j),sum)
    return out
 
def LoG_filter(image, sigma, size=None):
    # Generate LoG kernel
    if size is None:
        size = int(5 * sigma + 1) if sigma >= 1 else 7

    if size % 2 == 0:
        size += 1
    print(size)
    x, y = np.meshgrid(np.arange(-size//2+1, size//2+1), np.arange(-size//2+1, size//2+1))
    kernel = -(1/(np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(np.abs(kernel))

    # Perform convolution
    result = convolve(image, kernel)

    return result

    
img = cv2.imread('Lena.jpg',0)
w = img.shape[0]
h = img.shape[1]


filterLoG=np.array([[0.0,0.0,1.0,0.0,0.0],
        [0.0,1.0,2.0,1.0,0.0],
        [1.0,2.0,-16.0,2.0,1.0],
        [0.0,1.0,2.0,1.0,0.0],
        [0.0,0.0,1.0,0.0,0.0]])

Log = convolve(img,filterLoG)      
Log_norm = cv2.normalize(Log, 0, 255, cv2.NORM_MINMAX)
Log_norm = np.round(Log_norm).astype(np.uint8)
cv2.imshow('LoG',Log_norm)

Log2 = LoG_filter(img, 1)     
Log_norm2 = cv2.normalize(Log2, 0, 255, cv2.NORM_MINMAX)
Log_norm2 = np.round(Log_norm2).astype(np.uint8)
cv2.imshow('LoG2',Log_norm2)
#cv2.imwrite('H:\Fall 16\CVIP-CS573\HW\HW4\LoGApplied.jpg',Log)
plt.imshow(Log,cmap = 'gray')
plt.show()
emptyImage = np.zeros(shape=(w, h), dtype=int)
zeroCrossLog = np.zeros(shape=(w, h))

threshold = np.mean(img)
for i in range (1,w-1):
   for j in range (1,h-1):
       count=0
       local_region = [Log[i-1,j],Log[i+1,j],Log[i,j],Log[i,j-1],Log[i,j+1]]
       local_stddev = np.std(local_region)
       
       if (((Log[i-1,j]>0 and Log[i,j]<0) or (Log[i-1,j]<0 and Log[i,j]>0)) or ((Log[i+1,j]>0 and Log[i,j]<0) or (Log[i+1,j]<0 and Log[i,j]>0))) :
          count+=1
       elif (((Log[i,j-1]>0 and Log[i,j]<0) or (Log[i,j-1]<0 and Log[i,j]>0)) or ((Log[i,j+1]>0 and Log[i,j]<0) or (Log[i,j+1]<0 and Log[i,j]>0))):
          count+=1
       
       if(count<=0):
           zeroCrossLog[i,j]=0
       else:
           #print(local_region,local_stddev)
           if local_stddev > threshold:
               zeroCrossLog[i,j]=255

cv2.imshow("0cross", zeroCrossLog)

"""
LOGEdge= [[0.0 for x in range(ubh)] for y in range(ubw)]

for i in range(1, ubw):
        for j in range(1, ubh):
            LOGEdge[i][j] = abs(thresh[i][j][0] - zeroCrossLog[i][j])


sp.misc.toimage(LOGEdge).show()
#sp.misc.toimage(LOGEdge).save('H:\Fall 16\CVIP-CS573\HW\HW4\LOGEdges.jpg')
"""
cv2.waitKey(0)
cv2.destroyAllWindows()
