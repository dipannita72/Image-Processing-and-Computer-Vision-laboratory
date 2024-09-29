
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Create the input image
input_image = 255*np.array((
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    ), dtype="uint8")
 
# Construct the structuring element
kernel = np.array((
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]), dtype="uint8")
 

rate = 30
kernel = (kernel) * 255
kernel = np.uint8(kernel)
kernel = cv.resize(kernel, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
print(kernel.shape)
cv.imshow("kernel", kernel)
cv.imwrite("kernel.jpg", kernel)
cv.moveWindow("kernel", 700, 200)
input_image = cv.resize(input_image, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
print(input_image.shape)
cv.imshow("Original", input_image)
cv.moveWindow("Original", 0, 200)
cv.imwrite("input.jpg", input_image)

cv.waitKey(0)
cv.destroyAllWindows()