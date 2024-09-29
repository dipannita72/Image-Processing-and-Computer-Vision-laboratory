# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:43:47 2023

@author: USER
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
rate = 50
kernel = np.ones((150,150),np.uint8)
kernel2 = np.ones((50,50),np.uint8)

A = cv2.imread("intput1.jpg",0)
cv2.imshow("input",A)
F,A = cv2.threshold(A, 130, 255, cv2.THRESH_BINARY)
S = cv2.imread("kernel3.jpg",0)
cv2.imshow("kernel",S)
F,S = cv2.threshold(S, 130, 255, cv2.THRESH_BINARY)
print(A.shape)
print(S.shape)

A = A.astype(np.uint8)
S = S.astype(np.uint8)


W  = cv2.dilate(S,kernel,iterations = 1) 

cv2.imshow("window",W)
A_er_S = cv2.erode(A,S,iterations = 1)
cv2.imshow("windowAS",A_er_S)
#S_ = cv.copyMakeBorder(src=S, top=0, bottom=0, left=50, right=50,borderType= cv.BORDER_CONSTANT)#BORDER_WRAP, cv.BORDER_REFLECT  
WminusS = W-S
cv2.imshow("window2",WminusS)
A_c = cv2.bitwise_not(A)
A_cErWminusS = cv2.erode(A_c,WminusS,iterations = 1)

A_hit_S = cv2.bitwise_and(A_er_S,A_cErWminusS)
A_hit_S = cv2.dilate(A_hit_S,kernel2,iterations = 1) #cv.resize(A_hit_S, None , fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
cv2.imshow("hitormiss", A_hit_S)


cv2.waitKey(0)
cv2.destroyAllWindows()