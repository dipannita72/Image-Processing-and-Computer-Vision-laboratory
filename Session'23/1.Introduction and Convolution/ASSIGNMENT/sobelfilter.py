# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:30:45 2021

@author: snazm
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

image_original = cv2.imread('Lena.jpg', cv2.IMREAD_COLOR)

image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

cv2.imshow("input",image_gray)


rows, columns = np.shape(image_gray)

sobel_filtered_image = np.zeros(shape=(rows, columns))

sobel_filtered_x = np.zeros(shape=(rows, columns))
sobel_filtered_y = np.zeros(shape=(rows, columns))


sobel_y = np.array([[-1, -1, -1], 
                    [0, 0, 0], 
                    [1, 1, 1]])

sobel_x = np.array([[1, 0, -1], 
                    [1, 0, -1], 
                    [1, 0, -1]])


sobel_x = np.fliplr(sobel_x)
sobel_x = np.fliplr(sobel_x)
sobel_y = np.fliplr(sobel_y)
sobel_y = np.fliplr(sobel_y)

for i in range(rows - 2):
    for j in range(columns - 2):
        gx = np.sum(np.multiply(sobel_x, image_gray[i:i + 3, j:j + 3]))     
        sobel_filtered_x[i+1,j+1]=gx                                       
        gy = np.sum(np.multiply(sobel_y, image_gray[i:i + 3, j:j + 3]))   
        sobel_filtered_y[i+1,j+1]=gy                                        
        sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)     


cv2.normalize(sobel_filtered_x, sobel_filtered_x, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(sobel_filtered_y, sobel_filtered_y, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(sobel_filtered_image, sobel_filtered_image, 0, 255, cv2.NORM_MINMAX)
        

sobel_filtered_x = np.round(sobel_filtered_x).astype(np.uint8)
sobel_filtered_y = np.round(sobel_filtered_y).astype(np.uint8)
sobel_filtered_image = np.round(sobel_filtered_image).astype(np.uint8)
print(sobel_filtered_image)

cv2.imshow("sobel x",sobel_filtered_x)
cv2.imshow("sobel y",sobel_filtered_y)


cv2.imshow("sobel",sobel_filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()