# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:35:33 2024

@author: USER
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_derivative(x, y, sigma):
    return -x * np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**4),  -y * np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**4)

def apply_gaussian_derivative(image, sigma):
    rows, cols = image.shape
    gx = np.zeros_like(image, dtype=float)
    gy = np.zeros_like(image, dtype=float)
    pad =  int(3* sigma)
    for x in range(-pad, pad+ 1):
        for y in range(-pad, pad+ 1):
            weight_x, weight_y = gaussian_derivative(x, y, sigma)
            
            gx += np.roll(np.roll(image, x, axis=0), y, axis=1) * weight_x
            gy += np.roll(np.roll(image, x, axis=0), y, axis=1) * weight_y
    
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    cv2.normalize(gradient_magnitude, gradient_magnitude, 0, 255, cv2.NORM_MINMAX)
    gradient_magnitude = np.round(gradient_magnitude).astype(np.uint8)
    cv2.normalize(gx, gx, 0, 255, cv2.NORM_MINMAX)
    gx = np.round(gx).astype(np.uint8)
    cv2.normalize(gy , gy , 0, 255, cv2.NORM_MINMAX)
    gy  = np.round(gy ).astype(np.uint8)
    return gx,gy, gradient_magnitude


def adaptive_threshold(img, block_size, a,b ):
    # Convert image to grayscale
    p=block_size//2
    # Get dimensions of the image
    h, w = img.shape
    
    # Initialize output image
    thresholded = np.zeros_like(img)
    
    # Pad the image to handle border cases
    padded_img = np.pad(img, block_size // 2, mode='constant')
    
    # Iterate through each pixel of the image
    for i in range(p,h-p):
        for j in range(p,w-p):
            # Extract the local region around the pixel
            local_region = padded_img[i-p:i+p+1, j-p:j+p+1]
            print(local_region.shape)
            # Compute local mean and standard deviation
            local_mean = np.mean(local_region)
            local_stddev = np.std(local_region)
            print(local_mean, local_stddev,img[i,j])
            # Compute threshold value using mean and standard deviation
            threshold_a = a*local_mean
            threshold_b = b*local_stddev
            
            # Apply thresholding
            if img[i, j] > threshold_a and img[i, j] > threshold_b: 
                thresholded[i, j] = 255
            else:
                thresholded[i, j] = 0
    
    return thresholded

def main():
    # Read input image
    img = cv2.imread('buildings.jpg',0)
    
    # Define parameters for adaptive thresholding
    block_size = 5  # Size of the local region
    a = 2.3       # Constant subtracted from the mean
    b = 2
    sigma = 1
    # Apply adaptive thresholding
    gx,gy,gradient_magnitude = apply_gaussian_derivative(img, sigma)
    thresholded_img = adaptive_threshold(gradient_magnitude, block_size,a, b)
    
    # Display the original and thresholded images
    cv2.imshow('Original Image', img)
    cv2.imshow("Gradient", gradient_magnitude)
    cv2.imshow('Thresholded Image', thresholded_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
