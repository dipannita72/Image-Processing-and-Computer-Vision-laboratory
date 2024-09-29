#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:34:39 2023

@author: kanizfatema
"""


import cv2
import numpy as np
from math import sqrt,atan2,cos,sin


img_input = cv2.imread('f1.jpg')
#cv2.imshow("input",img_input)


out=np.zeros_like(img_input)
#out=np.full(img_input.shape,-1)


print(str(img_input.shape[1]/2)+" "+str(img_input.shape[0]/2))

amplitudex=(float)(input("Enter the amplitude  x axis: "))
amplitudey=(float)(input("Enter the amplitude  y axis: "))
#alpha=(float)(input("Enter the angle in degree: "))
periodx=(float)(input("Enter the period_length x axis: "))
periody=(float)(input("Enter the period_length y axis: "))
#centerx=(int)(input("Enter the center points x: "))
#centery=(int)(input("Enter the center points y: "))
centerx=img_input.shape[1]//2
centery=img_input.shape[0]//2
      

#alpha=alpha*np.pi/180

for j in range(img_input.shape[1]):
    for i in range(img_input.shape[0]):
        newx=int(i+(amplitudex*(np.sin((2*np.pi*j)/periodx))))
        newy=int(j+(amplitudey*(np.sin((2*np.pi*i)/periody))))
        if((0<=newx<img_input.shape[0]) and (0<=newy<img_input.shape[1])):
            out[i][j]=img_input[newx][newy]


               
#out2=cv2.resize(out,dsize=(out.shape[0],out.shape[1]),interpolation=cv2.INTER_LINEAR)    

out= cv2.bilateralFilter(out, 15, 75, 75)   
cv2.imwrite("ripple_2_75.jpg", out)



cv2.imshow("input",img_input)
cv2.imshow("output",out)
#cv2.imshow("output",out2)
#cv2.imshow("output_col",backtocol)            


cv2.waitKey(0)
cv2.destroyAllWindows() 