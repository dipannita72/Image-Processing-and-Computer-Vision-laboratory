import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def gaussian(sigma):
    flt = 3*sigma +1
    pad =flt//2
    gf = np.zeros((pad*2+1, pad*2+1))
    
    c = math.pi * 2 * sigma *sigma
    
    for i in range(-pad, pad+1):
        for j in range(-pad, pad+1):
            
            xy = -(i*i+j*j)/(2*sigma*sigma)
            
            d = (np.exp(xy))/c
            
            gf[i+pad][j+pad] = d
            
    summ = np.sum(gf)
    
    gf = gf/summ
    
    return gf

img=cv2.imread('noise.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("input image",img)
print(img.shape)

k= gaussian(1)

plt.imshow (k,cmap='gray')
plt.show()

#cv2.filter2D(img,-1,k)

for i in range(5,img.shape[0]-5):
    for j in range(5,img.shape[1]-5):
        sum=0.0
        for x in range(5):
            for y in range(5):
                a= img.item(i+x,j+y)
                b=k[x][y]*a
                sum=sum+b
                #print(sum)
        img.itemset((i,j),sum)

cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
img = np.round(img).astype(np.uint8)
cv2.imshow("gaussian Output image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()