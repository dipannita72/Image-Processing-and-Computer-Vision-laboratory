
# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dpc
from math import sqrt,exp
import matplotlib
import matplotlib.pyplot as plt




# take input
img_input = cv2.imread('pnois2.jpg', 0)
img = dpc(img_input)

image_size = img.shape[0] * img.shape[1]

# fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)
magnitude_spectrum_ac =  np.abs(ft_shift)
magnitude_spectrum =  30* np.log(np.abs(ft_shift)+1)

#mag, ang = cv2.cartToPolar(ft_shift[:,:,0],ft_shift[:,:,1])
ang = np.angle(ft_shift)

magnitude_spectrum_scaled = cv2.normalize(magnitude_spectrum, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

#<<<<<<-------------MOUSE EVENT-------------
point_list=[(261,261)] #(272,256),
'''
def Capture_Event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN:
        print(x,y)
        point_list.append((x,y))

#plt.title("Please select points from the input")
#im = plt.imshow(magnitude_spectrum_scaled, cmap='gray')
cv2.imshow('image', magnitude_spectrum_scaled)
cv2.setMouseCallback('image', Capture_Event)
#plt.show(block=True)
cv2.waitKey(0)
print(point_list)
'''
#-------------MOUSE EVENT------------->>>>>>>>>


D0=float (input("Enter the D0 "))


# remove noise from magnitude with ILPF
rows, cols = img.shape[0],img.shape[1]

base = np.ones((rows, cols), np.uint8)
cv2.imshow("filter1", base)
#print(point_list)
for x in range(0,rows):
    for y in range(0,cols):
        for xp in point_list:
             center = (xp[0],xp[1])
             #print(center)
             Dk = sqrt(( (x -center[0])**2) + ((y - center[1])**2))
             #print(x,y, Dk,end="s ")
             center2=((rows-center[0]),(cols-center[1]))
             #print(center, center2)
             Dkk =  sqrt(( (x -center2[0])**2) + ((y - center2[1])**2))
             #print(Dk,Dkk)
             if(Dkk < D0 or Dk < D0):   
                 print(y,x,rows-x-1,cols-y-1)
                 base[y][x] = 0
                 #base[rows-x-1][cols-y-1] = 0
             
             
#print(base[256,272])
filter_normalized = cv2.normalize(base, None, 0, 255.0,cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow("filter", filter_normalized)
cv2.imwrite("filter.jpg", filter_normalized)


float_magnitude_spectrum_ac=np.float32(magnitude_spectrum_ac)

notch_magnitude = np.multiply(filter_normalized,magnitude_spectrum_ac)
result_magnitude_spectrum_scaled = 30 * np.log(notch_magnitude+1)
result_magnitude_spectrum_scaled= cv2.normalize(result_magnitude_spectrum_scaled, None, 0, 255.0,cv2.NORM_MINMAX, dtype=cv2.CV_8U)

## phase add
notch_result_phase = np.multiply(notch_magnitude, np.exp(1j*ang))


# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(notch_result_phase)))
img_back_scaled = cv2.normalize(img_back, None, 0, 255.0,cv2.NORM_MINMAX, dtype=cv2.CV_8U)

## plot

cv2.imshow("input", img_input)
cv2.imshow("input spectrum", magnitude_spectrum_scaled)
cv2.imshow("Result Magnitude Spectrum",result_magnitude_spectrum_scaled)
cv2.imshow("Inverse transform",img_back_scaled)


cv2.imwrite('magnitude_spectrum_scaled.jpg',magnitude_spectrum_scaled)
cv2.imwrite("Result Magnitude Spectrum.jpg",result_magnitude_spectrum_scaled)
cv2.imwrite("Inverse transform_.jpg",img_back_scaled)



cv2.waitKey(0)
cv2.destroyAllWindows() 
