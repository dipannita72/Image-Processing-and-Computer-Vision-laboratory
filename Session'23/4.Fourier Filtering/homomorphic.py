

import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dpc
import math


def generate_filter(img,gammah,gammal,c,d0):
    x_center=img.shape[0]//2
    y_center=img.shape[1]//2
    filter_ = np.zeros((img.shape[0],img.shape[1]),np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = (i-x_center)
            y = (j-y_center)
            r = math.exp(-(((x**2+y**2)*c)/(d0**2)))
            r = (gammah-gammal)*(1-r)+gammal
            filter_[i][j] = r
    return filter_


def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')

# take input
img1 = cv2.imread('cor_img1.jpg', 0)
img = np.log1p(img1)

# fourier transform
ft = np.fft.fft2(img)
ft_shift_img = np.fft.fftshift(ft)
magnitude= np.abs(ft_shift_img)
ang = np.angle(ft_shift_img)

gh=1.5
gl=0.4
c=0.12
d0= 60

filter_ = generate_filter(img,gh,gl,c,d0)
filtered_img = ft_shift_img * filter_

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_img)))
#img_back = cv2.normalize(img_back, None, 0, 255,cv2.NORM_MINMAX).astype(np.uint8)
img_back=np.exp(img_back)-1
img_back_scaled = cv2.normalize(img_back, None, 0, 255,cv2.NORM_MINMAX).astype(np.uint8)



## plot
cv2.imshow("input", img1)

cv2.imshow("Inverse transform",img_back_scaled)

cv2.imwrite("output.jpg",img_back_scaled)

cv2.waitKey(0)
cv2.destroyAllWindows() 
'''
output = cv2.normalize(img_back, None, 0, 1.0,cv2.NORM_MINMAX).astype(np.uint8)

plt.imshow(magnitude,"gray")
plt.show()
cv2,imshow("hgj",magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()'''