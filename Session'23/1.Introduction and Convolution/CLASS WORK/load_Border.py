# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 23:37:12 2023

@author: USER
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('lena.jpg')
print(img.shape)


#%%
#gray = np.ones((128,128,3), dtype=np.uint8)
cv.imshow("Color",img)
kernel2 = (1/16) * np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]])
kernel1 = (1/25) * np.array([[1,1,1,1,1],
                   [1,1,1,1,1],
                   [1,1,1,1,1],
                   [1,1,1,1,1],
                   [1,1,1,1,1]])
b,g,r = cv.split(img)
#print(b.shape)
cv.imshow("Green",g)
cv.imshow("Red",r)
cv.imshow("Blue",b)
b2=cv.filter2D(src=b, ddepth=-1, kernel=kernel1,borderType=cv.BORDER_CONSTANT)
g2 =cv.filter2D(src=g, ddepth=-1, kernel=kernel1,borderType=cv.BORDER_CONSTANT)
r2 =cv.filter2D(src=r, ddepth=-1, kernel=kernel1,borderType=cv.BORDER_CONSTANT)
merged = cv.merge((b2,g2,r2))
cv.imshow("Merged",merged )

img2 = cv.cvtColor(img,cv.COLOR_RGB2HSV)
b,g,r = cv.split(img2)
b2=cv.filter2D(src=b, ddepth=-1, kernel=kernel1,borderType=cv.BORDER_CONSTANT)
g2 =cv.filter2D(src=g, ddepth=-1, kernel=kernel1,borderType=cv.BORDER_CONSTANT)
r2 =cv.filter2D(src=r, ddepth=-1, kernel=kernel1,borderType=cv.BORDER_CONSTANT)
merged2 = cv.merge((b2,g2,r2))
merged2 = cv.cvtColor(merged2,cv.COLOR_HSV2RGB)
cv.imshow("Merged2",merged2 )
s = merged - merged2
cv.imshow("subtract2",s )


blur = cv.blur(img,(5,5))
cv.imshow("blur",blur )
blur2 = cv.blur(img2,(5,5))
blur2 = cv.cvtColor(blur2,cv.COLOR_HSV2RGB)
cv.imshow("blur2",blur2 )
a= blur - blur2
cv.imshow("subtract",a )


#%%
image_bordered = cv.copyMakeBorder(src=img, top=25, bottom=25, left=25, right=25,borderType= cv.BORDER_CONSTANT)#BORDER_WRAP, cv.BORDER_REFLECT  
#print(image_bordered.shape)
#cv.imshow("Bordered",image_bordered )


#%%
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow("GRAY", gray)

#%%
im = plt.imread("Lena.jpg")
gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
#plt.title("GRAY IMG")
#plt.imshow(gray,"gray", vmin=0, vmax=255)
#plt.show()


cv.waitKey(0)
cv.destroyAllWindows()