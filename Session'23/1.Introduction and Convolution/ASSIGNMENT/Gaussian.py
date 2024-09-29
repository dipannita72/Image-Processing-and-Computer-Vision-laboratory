import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

img=cv2.imread('noise.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("input image",img)
print(img.shape)
kernel_size =5
k= np.ones((kernel_size,kernel_size),np.float32)
#cv2.filter2D(img,-1,k)
c=1/(2*3.1416)


for i in range(kernel_size):
    for j in range (kernel_size):
        a=np.power(i-(kernel_size//2),kernel_size//2)
        b=np.power(j-(kernel_size//2),kernel_size//2)
        k[i][j]=c*math.exp(-(a+b)/2)        


plt.imshow (k,cmap='gray')
plt.show()
print(k)
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

