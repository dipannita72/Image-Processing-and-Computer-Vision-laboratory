import numpy as np
import cv2
import math
img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
out=img.copy()

print(img.max())
print(img.min())
print(img.shape)


for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        a = img.item(i,j)
        out.itemset((i,j),255-a)
        
cv2.imshow('output image',out)

cv2.imshow('input image',img)


blur = cv2.GaussianBlur(img,(9,9),0)#(3, 3) is the kernel size, and 0 is the standard deviation value.
cv2.imshow('output image1',blur)
median = cv2.medianBlur(img,5)
kernel2 = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])
  

edge = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
cv2.imshow("edge",edge)

cv2.waitKey(0)
cv2.destroyAllWindows()
