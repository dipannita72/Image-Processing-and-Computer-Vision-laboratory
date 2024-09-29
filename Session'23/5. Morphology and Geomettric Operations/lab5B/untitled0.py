# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 00:12:26 2023

@author: USER
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

input_image = np.array((
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 0, 0, 255],
    [0, 0, 255, 255, 0, 0, 0, 0],
    [0, 0, 255, 255, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 255, 255, 0, 0, 255, 255, 255],
    [0,255, 0, 0, 0, 0, 255, 0],
    [0, 255, 0, 0, 0, 0, 255, 0]), dtype="uint8")
kernel1 = np.array((
        [1, 1, 1],
        [-1, 1, -1],
        [-1, 1, -1]), dtype="int")

kernel2 = np.array((
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]), dtype="uint8")

kernel3 =np.array((
    [0,1,1,0],
    [0,0,1,0],
    [0,0,1,0]), dtype = "uint8")

kernel4 =np.array((
    [0,0,0],
    [1,1,0],
    [1,0,0]), dtype = "uint8")
kernel = kernel1

output_image = cv.morphologyEx(input_image, cv.MORPH_HITMISS, kernel)


rate = 50
kernel = (kernel) * 255
kernel = np.uint8(kernel)
kernel = cv.resize(kernel, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
input_image = cv.resize(input_image, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
output_image = cv.resize(output_image, None , fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)



cv.imshow("Original", input_image)
cv.imshow("kernel", kernel)
cv.imshow("Hit or Miss", output_image)


cv.waitKey(0)
cv.destroyAllWindows()