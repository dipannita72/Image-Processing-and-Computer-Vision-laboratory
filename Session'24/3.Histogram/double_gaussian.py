

import cv2
import numpy as np
from numpy import random
import matplotlib.pyplot as plt


#function for normalization

def normal(arr):
   
    
    max = np.max(arr)
    min = np.min(arr)
   
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            
            arr[i][j] = ((arr[i][j]-min)/(max-min))*255
    print(arr)
    return arr
            
#function for matching histogram

        
#combine two guassian


def gaussian(miu, sigma, inp):
    gd = random.normal(miu, sigma, size=(inp.shape[0],inp.shape[1]))
    #print(gd)
    gd = np.round(gd).astype(int);
    #gd[gd>255]=255
    #gd[gd<0]= 0
 
    #gd = normal(gd)
    return gd
    
    
def match(img,img2):
    
    a1 = np.zeros(256, dtype="float32")
    p1 = np.zeros(256, dtype="float32")
    s1 = np.zeros(256, dtype="float32")
    
    a2 = np.zeros(256, dtype="float32")
    p2 = np.zeros(256, dtype="float32")
    s2 = np.zeros(256, dtype="float32")
    
    #input image frequency
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
        
            x = img[i][j]
            a1[x] = a1[x]+1
        
    ln1 = img.shape[0]*img.shape[1]
    
    
    #reference image frequency
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
        
            x = img2[i][j]
            a2[x] = a2[x]+1
        
    ln2 = img2.shape[0]*img2.shape[1]
    
    #%%
    
    #input image pdf
    for i in range(256):
        
        p1[i] = a1[i]/ln1
        
    #reference image pdf
    
    for i in range(256):
        
        p2[i] = a2[i]/ln2
        
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

gauss1 = gaussian(30, 8, inp)
gauss2 = gaussian(165, 20, inp)

con = np.concatenate((gauss1, gauss2))
plt.figure(3)
plt.hist(con.ravel(),256,[0,255])
plt.show()
plt.figure(12)
con = normal(con)
plt.hist(con.ravel(),256,[0,255])
plt.show()

print(con.shape)

final_output = match(inp, con)
plt.figure(4)
plt.hist(final_output.ravel(),256,[0,255])
plt.show()
plt.figure(5)     
plt.title("FInal Output")
plt.imshow(final_output,"gray")
plt.show()




    