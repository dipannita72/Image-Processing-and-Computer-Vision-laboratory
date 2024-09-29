#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:34:39 2023

@author: kanizfatema
"""


import cv2
import numpy as np
from math import sqrt,atan2,cos,sin


img_input = cv2.imread('checker.jpg', 0)
#cv2.imshow("input",img_input)


out=np.zeros_like(img_input)
#out=np.full(img_input.shape,-1)


print(str(img_input.shape[1]/2)+" "+str(img_input.shape[0]/2))

r_max=(int)(input("Enter the radius: "))
alpha=(float)(input("Enter the angle in degree: "))
#centerx=(int)(input("Enter the center points x: "))
#centery=(int)(input("Enter the center points y: "))
centerx=img_input.shape[1]//2
centery=img_input.shape[0]//2
      

alpha_rad=alpha*np.pi/180

for j in range(img_input.shape[1]):
    for i in range(img_input.shape[0]):
        dy=j-centery
        dx=i-centerx
        r=sqrt(dx*dx + dy*dy )
        if (r <= r_max):
            angle_offset = alpha_rad * (r_max - r) / r_max
            bita=atan2(dy,dx)+angle_offset
            newx=int(centerx+(r*cos(bita)))
            newy=int(centery+(r*sin(bita)))
            if((0<=newx<img_input.shape[1]) and (0<=newy<img_input.shape[0])):
                out[newx][newy]=img_input[i][j]
        else:
             out[i][j]=img_input[i][j]
'''
for j in range(out.shape[1]):
    for i in range(out.shape[0]):
        dy=j-centery
        dx=i-centerx
        r=sqrt(dx*dx + dy*dy )
        if (r <= r_max):
            if(out[i][j]==-1):
                sum=0
                point_list=[]
                for a in range (-1,2):
                    for b in range (-1,2):
                        if(a!=0 and b!=0):
                            if(0<=a+i<out.shape[1] and 0<=b+j<out.shape[0]):
                                #sum=sum+out[a+i][b+j]
                                point_list.append(out[a+i][b+j])
                point_list.sort()
                middleIndex = (len(point_list))//2
                out[i][j]=point_list[middleIndex]
        else:
            if(out[i][j]==-1):
                out[i][j]=0

out = out.astype(np.uint8)

'''
cv2.imwrite("Checker_twriled__150.jpg", out)

out=cv2.medianBlur(out,7) 
cv2.imshow("input",img_input)
cv2.imshow("output",out)
              


cv2.waitKey(0)
cv2.destroyAllWindows() 