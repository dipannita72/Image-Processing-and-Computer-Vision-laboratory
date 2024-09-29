import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def gaussian_derivative(x, y, sigma):
    return -x * np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**4),  -y * np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**4)

def derivative_kernel(sigma):
    pad =  int(3* sigma)
    g_x = np.zeros((pad*2+1, pad*2+1))
    g_y = np.zeros((pad*2+1, pad*2+1))
    print(pad)
    print(g_x.shape)
    for x in range(-pad, pad+ 1):
        for y in range(-pad, pad+ 1):
            weight_x, weight_y = gaussian_derivative(x, y, sigma)
            g_x[y+pad][x+pad] = weight_x
            g_y[y+pad][x+pad] = weight_y
    sum1 = np.sum(g_x)
    sum2 = np.sum(g_y)
    g_x = g_x/sum1
    g_y = g_y/sum2
    return g_x, g_y
            

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

def global_thresholding(image):
    """
    Apply global thresholding to detect edges.
    """
    T = np.mean(image)
    print(T)
    # Iterate until convergence
    while True:
        # Segment the image using T
        G1 = image > T
        G2 = image <= T

        # Compute average grey levels of pixels in G1 and G2
        μ1 = np.mean(image[G1])
        μ2 = np.mean(image[G2])

        # Update threshold
        new_T = (μ1 + μ2) / 2

        # Check for convergence
        if np.abs(T - new_T) < 1e-4:
            break

        T = new_T
    
    print("threshold"+str(T))
    _, binary_edge = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
    return binary_edge
sigma = 2
k_x, k_y= derivative_kernel(sigma)
plt.figure(1)
plt.imshow (k_x,cmap='gray')
plt.figure(2)
plt.imshow (k_y,cmap='gray')

# Load the image
image = cv2.imread('lines5.jpg', cv2.IMREAD_GRAYSCALE)
print(image.min())
print(image.max())
hist = cv2.calcHist(image,[0],None,[256],[0,255]) 

plt.figure(3)
plt.plot(hist) 
plt.show() 
# Apply Gaussian derivative filter

gx,gy,gradient_magnitude = apply_gaussian_derivative(image, sigma)
hist2 = cv2.calcHist(gradient_magnitude,[0],None,[25],[0,256]) 
plt.figure(4)
plt.plot(hist2) 
plt.show()
binary_edge = global_thresholding(gradient_magnitude)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Gradient x', gx.astype(np.uint8))
cv2.imshow('Gradient y', gy.astype(np.uint8))
cv2.imshow('Gradient Magnitude', gradient_magnitude.astype(np.uint8))
cv2.imshow('Binary Edge Image', binary_edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
