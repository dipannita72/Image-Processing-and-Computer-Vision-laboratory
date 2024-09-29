
# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dpc
from math import sqrt,exp
import matplotlib
import matplotlib.pyplot as plt


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
magnitude_spectrum =  30* np.log(np.abs(magnitude_spectrum_ac)+1)

#mag, ang = cv2.cartToPolar(ft_shift[:,:,0],ft_shift[:,:,1])
ang = np.angle(ft_shift)

magnitude_spectrum_scaled = cv2.normalize(magnitude_spectrum, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

#<<<<<<-------------MOUSE EVENT-------------
point_list=[(272,256),(261,261)] 
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


D0=5 #float (input("Enter the D0 "))
n = 2

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

notch_magnitude = np.multiply(base,magnitude_spectrum)
#result_magnitude_spectrum_scaled = 20 * np.log(notch_magnitude+1)
result_magnitude_spectrum_scaled=min_max_normalize(notch_magnitude)
result_magnitude_spectrum_ac = np.exp(notch_magnitude / 30) - 1
## phase add
notch_result_phase = np.multiply(result_magnitude_spectrum_ac, np.exp(1j*ang))


# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(notch_result_phase)))
img_back_scaled = min_max_normalize(img_back)

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
