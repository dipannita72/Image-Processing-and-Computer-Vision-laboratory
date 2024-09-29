# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:22:04 2023

@author: USER
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
rate = 50
kernel = np.ones((3,3),np.uint8)


S = np.zeros((7,7),np.uint8)
S[3,1:6] = 255
S[1:6, 2:5] = 255
A = np.zeros((17,17),np.uint8)
A[0:7,0:7]=S
A[10:17,10:17] = S
A[13,16] = 255
A_ = cv.resize(A, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST) 
S_= cv.resize(S, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST) 
cv.imwrite("kernel.jpg", S)
cv.imwrite("img.jpg", A)
print(S)
print(A)
#cv.imshow("S1",A)
#cv.imshow("S2",S)
W  = cv.dilate(S,kernel,iterations = 1) 
plt.imshow(A,cmap="gray")
plt.show(block = True)

plt.imshow(S,cmap="gray")
plt.show(block = True)
plt.imshow(W,cmap="gray")
plt.show(block = True)

A_er_S = cv.erode(A,S,iterations = 1)
#cv.imshow("S2",A_er_S)
plt.imshow(A_er_S ,cmap="gray")
plt.show(block = True)

WminusS = W-S
plt.imshow(WminusS,cmap="gray")
plt.show(block = True)

A_c = cv.bitwise_not(A)
plt.imshow(A_c ,cmap="gray")
plt.show(block = True)

A_cErWminusS = cv.erode(A_c,WminusS,iterations = 1)
plt.imshow(A_cErWminusS,cmap="gray")
plt.show(block = True)

A_hit_S = cv.bitwise_and(A_er_S,A_cErWminusS)
plt.imshow(A_hit_S,cmap="gray")
plt.show(block = True)

plt.show(block = True)
#cv.imshow("Dilation", W)
cv.waitKey(0)
cv.destroyAllWindows()