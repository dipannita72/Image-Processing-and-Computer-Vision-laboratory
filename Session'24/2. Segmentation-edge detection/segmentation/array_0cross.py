# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:44:11 2024

@author: USER
"""

import cv2
import numpy as np
import scipy as sp
import scipy.signal as sg
import matplotlib.pyplot as plt


Log = np.array([
    [1, 1, 1, -1, 1],
    [3, 3, -3, 3, 4],
    [3, 4, 4, 3, 1],
    [1, -1, 1, 1, 2],
    [1, 2, -2, 1, 2]
])
#img = cv2.imread('Lena.jpg',0)
w = Log.shape[0]
h = Log.shape[1]

emptyImage = np.zeros(shape=(w, h), dtype=int)
zeroCrossLog = np.zeros(shape=(w, h))

threshold = 2
print(threshold)
print(Log)
for i in range(1, w-1):
    for j in range(1, h-1):
        count = 0
        local_region = [Log[i-1, j], Log[i+1, j],
                        Log[i, j], Log[i, j-1], Log[i, j+1]]
        
        if (((Log[i-1, j] > 0 and Log[i, j] < 0) or (Log[i-1, j] < 0 and Log[i, j] > 0)) or ((Log[i+1, j] > 0 and Log[i, j] < 0) or (Log[i+1, j] < 0 and Log[i, j] > 0))):
            count += 1
        elif (((Log[i, j-1] > 0 and Log[i, j] < 0) or (Log[i, j-1] < 0 and Log[i, j] > 0)) or ((Log[i, j+1] > 0 and Log[i, j] < 0) or (Log[i, j+1] < 0 and Log[i, j] > 0))):
            count += 1

        if(count <= 0):
            zeroCrossLog[i, j] = 0
        else:
            zeroCrossLog[i, j] = Log[i, j]

print(zeroCrossLog)

for i in range(1, w-1):
    for j in range(1, h-1):
        local_region = [Log[i-1, j], Log[i+1, j],
                        Log[i, j], Log[i, j-1], Log[i, j+1]]
        local_stddev = np.std(local_region)
        if local_stddev > threshold:
            
            print(local_region, local_stddev)
            zeroCrossLog[i, j] = 255
        else:
            zeroCrossLog[i, j] = 0
print(zeroCrossLog)


cv2.waitKey(0)
cv2.destroyAllWindows()
