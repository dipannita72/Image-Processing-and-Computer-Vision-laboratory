import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from operator import add
from functools import reduce
import cv2
def split4(image):
    half_split = np.array_split(image, 2)
    res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
    return reduce(add, res)



def concatenate4(north_west, north_east, south_west, south_east):
    top = np.concatenate((north_west, north_east), axis=1)
    bottom = np.concatenate((south_west, south_east), axis=1)
    return np.concatenate((top, bottom), axis=0)


def calculate_mean(img):
    return np.mean(img, axis=(0, 1))

img = cv2.imread('lena.jpg')
split_img = split4(img)
print(split_img[0].shape)


fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(split_img[0])
axs[0, 1].imshow(split_img[1])
axs[1, 0].imshow(split_img[2])
axs[1, 1].imshow(split_img[3])

full_img = concatenate4(split_img[0], split_img[1], split_img[2], split_img[3])
plt.figure(3)
plt.imshow(full_img)
plt.show()

plt.figure(2)
means = np.array(list(map(lambda x: calculate_mean(x), split_img))).astype(int).reshape(2,2,3)
print(means)
plt.imshow(means)
plt.show()


