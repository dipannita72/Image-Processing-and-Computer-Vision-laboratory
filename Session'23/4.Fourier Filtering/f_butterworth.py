#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:30:26 2023

@author: kanizfatema
"""


# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dpc
from math import sqrt,exp
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# Normalization Function
def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')



# take input
img_input = cv2.imread('two_noise.jpeg', 0)
img = dpc(img_input)

image_size = img.shape[0] * img.shape[1]

# fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)
magnitude_spectrum_ac =  np.abs(ft_shift)
magnitude_spectrum =  30* np.log(np.abs(ft_shift))

#mag, ang = cv2.cartToPolar(ft_shift[:,:,0],ft_shift[:,:,1])
ang = np.angle(ft_shift)

magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

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
im = plt.imshow(magnitude_spectrum_scaled, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)

print(point_list)

#-------------MOUSE EVENT------------->>>>>>>>>


D0=float (input("Enter the D0 "))
n=int (input("Enter the order n "))


# remove noise from magnitude with BLPF
rows, cols = img.shape[0],img.shape[1]

base = np.zeros((rows, cols), np.float32)


for x in range(rows):
    for y in range(cols):
        for xp in point_list:
             center = (xp[0],xp[1])
             
             Dk = sqrt(( (x -center[0])**2) + ((y - center[1])**2))
             center2=((rows-center[0]-1),(cols-center[1]-1))
             Dkk =  sqrt(( (x -center2[0])**2) + ((y - center2[1])**2))
             #if(Dkk > D0 or Dk > D0):   
             if Dk ==0:
                 Dk = 1
             if Dkk ==0:
                 Dkk = 1    
             base[y][x] += (1.0/(1.0+(D0/Dk)**(2*n))) * (1.0/(1.0+(D0/Dkk)**(2*n)))
               #base[rows-x-1][cols-y-1]+= (1.0/(1.0+(D0/Dkk)**(2*n))) #*(1.0/(1.0+(D0/Dk)**(2*n)))
print(center)
filter_normalized = cv2.normalize(base, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.imshow("filter", filter_normalized)
cv2.imwrite("filter.jpg", filter_normalized)


float_magnitude_spectrum_ac=np.float32(magnitude_spectrum_ac)

notch_magnitude = np.multiply(filter_normalized,magnitude_spectrum_ac)
result_magnitude_spectrum_scaled = 20 * np.log(notch_magnitude+1)
result_magnitude_spectrum_scaled=min_max_normalize(result_magnitude_spectrum_scaled)

## phase add
notch_result_phase = np.multiply(notch_magnitude, np.exp(1j*ang))


# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(notch_result_phase)))
img_back_scaled = min_max_normalize(img_back)

## plot

cv2.imshow("input", img_input)
cv2.imshow("input spectrum", magnitude_spectrum_scaled)
cv2.imshow("Result Magnitude Spectrum",result_magnitude_spectrum_scaled)
cv2.imshow("Inverse transform",img_back_scaled)

'''
cv2.imwrite('magnitude_spectrum_scaled.jpg',magnitude_spectrum_scaled)
cv2.imwrite("Result Magnitude Spectrum.jpg",result_magnitude_spectrum_scaled)
cv2.imwrite("Inverse transform.jpg",img_back_scaled)
'''


cv2.waitKey(0)
cv2.destroyAllWindows() 
