# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:39:16 2023

@author: USER
"""

import cv2
import numpy

def gaussian(x,sigma):
    return (1.0/(2*numpy.pi*(sigma**2)))*numpy.exp(-(x**2)/(2*(sigma**2)))

def epanechnikov(distance,k_size):
    e = 1-(distance / k_size)**2
    if e< 0:
        return 0
    else:
        return e

def epan(intensity):
    e = 1 - (intensity/255)**2
    if e< 0:
        return 0
    else:
        return e

def sigmoid(x,sigma):
    return (2.0/numpy.pi)*(1/(numpy.exp(sigma) + numpy.exp(-sigma)))


def distance(x1,y1,x2,y2):
    return numpy.sqrt(numpy.abs((x1-x2)**2-(y1-y2)**2))

def bilateral_filter(image, diameter, sigma_i, sigma_s):
    new_image = numpy.zeros(image.shape)
    pad = diameter // 2
    for row in range(pad, len(image) -pad ):
        for col in range(pad,  len(image[0]) - pad ):
            wp_total = 0
            filtered_image = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x =row - (diameter/2 - k)
                    n_y =col - (diameter/2 - l)
                    if n_x >= len(image[1]):
                        n_x -= len(image[1])
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])
                    substract = int(image[int(n_x)][int(n_y)]) - int(image[row][col])
                    gi = gaussian(substract, sigma_i)
                    #gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                    wp = gi #* gs
                    filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            new_image[row][col] = int(numpy.round(filtered_image))
    return new_image


def bilateral_diffkernel(image, diameter, sigma_i):
    new_image = numpy.zeros(image.shape)
    pad = diameter // 2
    for row in range(pad, len(image) -pad ):
        for col in range(pad,  len(image[0]) - pad):
            wp_total = 0
            filtered_image = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x =row - (diameter//2 - k)
                    n_y =col - (diameter//2 - l)
                    if n_x >= len(image[1]):
                        n_x -= len(image[1])
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])
                    print(n_x,n_y,row,col)
                    substract = int(image[int(n_x)][int(n_y)]) - int(image[row][col])
                    gi = gaussian(substract, sigma_i)
                    rad = distance(n_x, n_y, row, col)
                    gs = epanechnikov(rad,diameter//2)
                    wp = gi * gs
                    filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            new_image[row][col] = int(numpy.round(filtered_image))
    return new_image
   
def bilateral_test(image, diameter, sigma_s):
    new_image = numpy.zeros(image.shape)
    pad = diameter // 2
    for row in range(pad, len(image) -pad ):
        for col in range(pad,  len(image[0]) - pad):
            wp_total = 0
            filtered_image = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x =row - (diameter//2 - k)
                    n_y =col - (diameter//2 - l)
                    if n_x >= len(image[1]):
                        n_x -= len(image[1])
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])
                    print(n_x,n_y,row,col)
                    substract = int(image[int(n_x)][int(n_y)]) - int(image[row][col])
                    #gi = gaussian(substract, sigma_i)
                    
                    gi = epan(substract)
                    rad = distance(n_x, n_y, row, col)
                    gs = gaussian(rad,sigma_s)
                    
                    
                    wp = gi * gs
                    filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            new_image[row][col] = int(numpy.round(filtered_image))
    return new_image
    
kernel_size = 7
pad = kernel_size // 2
image = cv2.imread("cube.png",0)
cv2.imwrite("cube_gray.jpg",image)
print(image.shape)
img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, None , value =0)

'''
filtered_image_OpenCV = cv2.bilateralFilter(image, kernel_size, 80.0, 20.0)
filtered_image_OpenCV = cv2.normalize(filtered_image_OpenCV, None, 0, 255,cv2.NORM_MINMAX).astype(numpy.uint8)
cv2.imshow("bilateral",filtered_image_OpenCV)
cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
'''

image_own = bilateral_filter(img, kernel_size, 80.0, 20.0)
image_own = cv2.normalize(image_own, None, 0, 255,cv2.NORM_MINMAX).astype(numpy.uint8)
image_own = image_own[pad:-pad,pad:-pad]
cv2.imshow("bilateral_own",image_own)
cv2.imwrite("filtered_image_own_box.png", image_own)

'''
image_epan = bilateral_test(img, kernel_size, 0.2)
image_epan = cv2.normalize(image_epan, None, 0, 255,cv2.NORM_MINMAX).astype(numpy.uint8)
image_epan = image_epan[pad:-pad,pad:-pad]
cv2.imshow("bilateral__epan",image_epan)
cv2.imwrite("filtered_epan_test_.2_3.jpg",image_epan)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()
