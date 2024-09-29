


import cv2
import numpy as np
from math import sqrt,atan2,cos,sin


img_input = cv2.imread('peppers.png')
#cv2.imshow("input",img_input)


out=np.zeros_like(img_input)
#out=np.full(img_input.shape,-1)


print(str(img_input.shape[1]/2)+" "+str(img_input.shape[0]/2))

amplitude=(float)(input("Enter the amplitude: "))

period_l=(float)(input("Enter the period_length: "))
#centerx=(int)(input("Enter the center points x: "))
#centery=(int)(input("Enter the center points y: "))
centerx=img_input.shape[1]//2
centery=img_input.shape[0]//2
      



for j in range(img_input.shape[1]):
    for i in range(img_input.shape[0]):
        dy=j-centery
        dx=i-centerx
        r=sqrt(dx*dx + dy*dy )
        angle_offset = amplitude * np.sin(2 * np.pi * (1/period_l) * r)
        theta=atan2(dy,dx)+angle_offset
        newx=int(centerx+(r*cos(theta)))
        newy=int(centery+(r*sin(theta)))
        if((0<=newx<img_input.shape[0]) and (0<=newy<img_input.shape[1])):
            out[i][j]=img_input[newx][newy]
out2=out
                
#out = cv2.bilateralFilter(out, 15, 75, 75)     

cv2.imwrite("Checker_bilateral.jpg", out)

cv2.imshow("input",img_input)
cv2.imshow("output",out)           


cv2.waitKey(0)
cv2.destroyAllWindows() 