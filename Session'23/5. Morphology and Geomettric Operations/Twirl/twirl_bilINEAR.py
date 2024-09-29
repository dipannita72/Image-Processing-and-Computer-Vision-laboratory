
import cv2
import numpy as np
from math import sqrt,atan2,cos,sin


img_input = cv2.imread('peppers.png')
#cv2.imshow("input",img_input)


out=np.zeros_like(img_input)
#out=np.full(img_input.shape,-1)


print(str(img_input.shape[1]/2)+" "+str(img_input.shape[0]/2))

r_max=(int)(input("Enter the radius: "))
alpha=(float)(input("Enter the angle in degree: "))
#centerx=(int)(input("Enter the center points x: "))
#centery=(int)(input("Enter the center points y: "))
centerx=img_input.shape[1]//2
centery=img_input.shape[0]//2
      

alpha_rad=alpha*np.pi/180

for j in range(img_input.shape[1]):
    for i in range(img_input.shape[0]):
        dy=j-centery
        dx=i-centerx
        r=sqrt(dx*dx + dy*dy )
        if (r <= r_max):
            angle_offset = alpha_rad * (r_max - r) / r_max
            bita=atan2(dy,dx)+angle_offset
            newx=int(centerx+(r*cos(bita)))
            newy=int(centery+(r*sin(bita)))
            if((0<=newx<img_input.shape[1]) and (0<=newy<img_input.shape[0])):
                out[i][j]=img_input[newx][newy]
        else:
             out[i][j]=img_input[i][j]

cv2.imwrite("Checker__.jpg", out)

#out = cv2.bilateralFilter(out, 15, 75, 75) #cv2.medianBlur(out,7) 
cv2.imshow("input",img_input)
cv2.imshow("output",out)
              


cv2.waitKey(0)
cv2.destroyAllWindows() 