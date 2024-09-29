

import cv2
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.special import factorial

#function for normalization

def normal(arr):
   
    
    max = np.max(arr)
    min = np.min(arr)
   
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            
            arr[i][j] = ((arr[i][j]-min)/(max-min))*255
    print(arr)
    return arr
            

def erlang_distribution(x, k, lambd):
    """Compute the Erlang distribution for the given parameters."""
    coefs = ((lambd**k) * (x**(k - 1))) / factorial(k - 1)
    return coefs * np.exp(-lambd * x)
    
    
def match(img,p2):
    
    a1 = np.zeros(256, dtype="float32")
    p1 = np.zeros(256, dtype="float32")
    s1 = np.zeros(256, dtype="float32") 
    s2 = np.zeros(256, dtype="float32")
    
    #input image frequency
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
        
            x = img[i][j]
            a1[x] = a1[x]+1
        
    ln1 = img.shape[0]*img.shape[1]
    for i in range(256):
        
        p1[i] = a1[i]/ln1
        
    #%%
    
    #input image sum
    sum = 0
    for i in range(256):
        sum = sum + p1[i]
        
        s1[i] = round(sum*255)
        
    #reference image sum
    
    sum = 0
    
    for i in range(256):
        
        sum = sum + p2[i]
        
        s2[i] = round(sum*255)
    #%%
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            k = img[i][j]
            c = 0
            for m in range(256):
                if (s2[m]>=s1[k]):
                    c=m
                    break
            
            img[i][j] = c
        
        
    return img


#input image
inp = cv2.imread('histogram.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure(1)
plt.title("Input")
plt.imshow(inp,"gray")
plt.show()
plt.figure(2)
plt.hist(inp.ravel(),256,[0,255])
plt.show()
#total length of input image
z = inp.shape[0]*inp.shape[1]
x = np.linspace(0, 255,256)
erg = erlang_distribution(x, k = 80, lambd = 0.6)
erg1 = np. round(erg, 2)
plt.figure(3)
plt.plot(x, erg, label=f'Erlang distribution (k={3}, λ={0.5})')
plt.show()



print("ads",erg)
final_output = match(inp, erg)
cv2.imshow("output", final_output)
plt.figure(4)
plt.hist(final_output.ravel(),256,[0,255])
plt.show()
plt.figure(5)     
plt.title("FInal Output")
plt.imshow(final_output,"gray")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()


    