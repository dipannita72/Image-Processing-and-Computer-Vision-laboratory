import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def gaussian(sigma_x,sigma_y):
    flt = int(5*sigma_x) +1
    pad =flt//2
    gf = np.zeros((pad*2+1, pad*2+1))
    g_x = np.zeros((pad*2+1, pad*2+1))
    g_y = np.zeros((pad*2+1, pad*2+1))
    c = math.pi * 2 * sigma_x *sigma_y
    
    for i in range(-pad, pad+1):
        for j in range(-pad, pad+1):
            
            xy = -(i*i+j*j)/(2*sigma_x*sigma_y)
            d = (np.exp(xy))/c
            y_dev = (-i/sigma_x**2)*d #title="y-derivative of 2D-Gauss"
            x_dev= (-j/sigma_y**2)*d #title="x-derivative of 2D-Gauss"
            gf[i+pad][j+pad] = d
            g_x[i+pad][j+pad] = x_dev
            g_y[j+pad][i+pad] = y_dev

    sum1 = np.sum(g_x)
    sum2 = np.sum(g_y)
    g_x = g_x/sum1
    g_y = g_y/sum2
    summ = np.sum(gf)
    gf = gf/summ
    return gf, g_x, g_y

def convolution(img, kernel):
    w = kernel.shape[0]
    h = kernel.shape[1]
    
    for i in range(w,img.shape[0]-w):
        for j in range(h,img.shape[1]-h):
            sum=0.0
            for x in range(kernel.shape[0]):
                for y in range(kernel.shape[1]):
                    a= img.item(i+x,j+y)
                    b=k[x][y]*a
                    sum=sum+b
                    #print(sum)
            img.itemset((i,j),sum)

    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img = np.round(img).astype(np.uint8)
    return img
    
img=cv2.imread('noise.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("input image",img)
print(img.shape)

k,k_x, k_y= gaussian(1,1)
plt.figure(1)
plt.imshow (k_x,cmap='gray')
plt.figure(2)
plt.imshow (k_y,cmap='gray')
plt.figure(3)
plt.imshow (k,cmap='gray')
plt.show()

#cv2.filter2D(img,-1,k)
out = convolution(img, k)
cv2.imshow("gaussian Output image",out)

outx = convolution(img, k_x)
cv2.imshow("gaussianx Output image",outx)

outy = convolution(img, k_y)
cv2.imshow("gaussiany Output image",outy)

cv2.waitKey(0)
cv2.destroyAllWindows()