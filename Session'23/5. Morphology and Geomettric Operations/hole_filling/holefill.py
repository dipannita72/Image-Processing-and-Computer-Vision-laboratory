# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:08:00 2023

@author: USER
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Read image
im_in = cv2.imread("hole2.jpg", cv2.IMREAD_GRAYSCALE)
th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)
h, w = im_in.shape[:2]
# Copy the thresholded image.
im_floodfill = im_in.copy()
floodfill = np.zeros((h,w),np.uint8)
#cv2.imshow("input",im_in)
#<<<<<<-------------MOUSE EVENT-------------
point_list=[]

def onclick(event):
    #global x, y
    ax = event.inaxes
    if ax is not None:
        print(str(event.x)+" "+str(event.y))
        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int(round(x))
        y = int(round(y))
        #print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              #(event.button, event.x, event.y, x, y))
        point_list.append((x,y))

plt.title("Please select points from the input")
im = plt.imshow(im_in, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)

print(point_list)

#---------------------------------------------------------------

im_floodfill_inv = cv2.bitwise_not(im_in)
im_out = im_in
mask = np.ones((50,50), np.uint8)
mask = cv2.getStructuringElement(cv2.MORPH_CROSS,(50,50))

for a in point_list:
    floodfill[a[1],a[0]]=255
    temp = floodfill
    while(1):
    
        
        im_floodfill = cv2.dilate(floodfill,mask,iterations = 1) 
        floodfill = im_floodfill & im_floodfill_inv
        if(np.all(temp == floodfill)):
            print("true")
            break
        temp = floodfill
    im_out = im_out | floodfill


# Display images.
plt.imshow(mask, cmap='gray')
plt.show(block=True)

cv2.imshow("out",im_out)
cv2.imwrite("holefill1.jpg",im_out)
cv2.waitKey(0)
cv2.destroyAllWindows()