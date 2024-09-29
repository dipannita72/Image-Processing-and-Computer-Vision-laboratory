


import cv2
import numpy as np
from math import sqrt,atan2,cos,sin


img_input = cv2.imread('s.jpg')
#cv2.imshow("input",img_input)


out=np.zeros_like(img_input)
#out=np.full(img_input.shape,-1)


print(str(img_input.shape[1]/2)+" "+str(img_input.shape[0]/2))

amplitude = (float)(input("Enter the amplitude: "))
tao_1 = (float)(input("Enter the wave_length: "))
tao_2 = (float)(input("Enter the wave_length: "))
#centerx=(int)(input("Enter the center points x: "))
#centery=(int)(input("Enter the center points y: "))
centerx=img_input.shape[1]//2
centery=img_input.shape[0]//2
      

#alpha=alpha*np.pi/180

for j in range(img_input.shape[1]):
    for i in range(img_input.shape[0]):
        dy=j-centery
        dx=i-centerx
        angle_x = amplitude * np.sin(2 * np.pi * (1/tao_1) * dx)
        angle_y = amplitude * np.sin(2 * np.pi * (1/tao_2) * dy)
        newx=int(i+angle_x)
        newy=int(j+angle_y)
        if((0<=newx<img_input.shape[0]) and (0<=newy<img_input.shape[1])):
            out[i][j]=img_input[newx][newy]

               
out = cv2.bilateralFilter(out, 15, 75, 75)   #bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType)


cv2.imwrite("Checker_.jpg", out)

#backtocol = cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)

cv2.imshow("output",out)
#cv2.imshow("output_col",backtocol)            


cv2.waitKey(0)
cv2.destroyAllWindows() 