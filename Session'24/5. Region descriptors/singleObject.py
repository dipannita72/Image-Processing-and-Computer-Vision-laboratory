import cv2
import numpy as np

def find_border_extreme_points(border_image):
    # Find non-zero indices in the border image
    nonzero_indices = np.argwhere(border_image != 0)

    # Find leftmost, rightmost, topmost, and bottommost points
    leftmost = tuple(nonzero_indices[np.argmin(nonzero_indices[:, 1])])
    rightmost = tuple(nonzero_indices[np.argmax(nonzero_indices[:, 1])])
    topmost = tuple(nonzero_indices[np.argmin(nonzero_indices[:, 0])])
    bottommost = tuple(nonzero_indices[np.argmax(nonzero_indices[:, 0])])

    return leftmost, rightmost, topmost, bottommost

def calculate_axes_lengths(leftmost, rightmost, topmost, bottommost):
    # Calculate major axis length
    major_axis_length = np.sqrt((rightmost[0] - leftmost[0]) ** 2 + (rightmost[1] - leftmost[1]) ** 2)

    # Calculate minor axis length
    minor_axis_length = np.sqrt((topmost[0] - bottommost[0]) ** 2 + (topmost[1] - bottommost[1]) ** 2)

    return major_axis_length/2, minor_axis_length/2


image = cv2.imread('star.png', cv2.IMREAD_GRAYSCALE)

# Apply binary thresholding
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)
eroded = cv2.erode(binary, kernel, iterations=1)
border = binary - eroded
perimeter = np.count_nonzero(border)
area = np.count_nonzero(binary)
print(perimeter, area)
circularity = (4 * np.pi * area) / (perimeter ** 2)
 
compactness = area / (np.pi * (perimeter ** 2) / 4)


moments = cv2.moments(border)

# Calculate centroid (center of mass)
cx = moments['m10'] / moments['m00']
cy = moments['m01'] / moments['m00']

# Centralize moments
central_moments = cv2.moments(border, True)
# Calculate covariance matrix
cov_matrix = np.array([[central_moments['mu20'], central_moments['mu11']],
                       [central_moments['mu11'], central_moments['mu02']]])
# Calculate eigenvalues and eigenvectors of the covariance matrix
_, eigenvalues, _ = np.linalg.svd(cov_matrix)
# Calculate eccentricity as the ratio of major and minor axes lengths
eccentricity = np.sqrt(1 - min(eigenvalues) / max(eigenvalues))
print(eccentricity)


leftmost, rightmost, topmost, bottommost = find_border_extreme_points(border)
major_axis_length, minor_axis_length = calculate_axes_lengths(leftmost, rightmost, topmost, bottommost)
print("Major axis length:", major_axis_length)
print("Minor axis length:", minor_axis_length)

eccentricity = np.sqrt(major_axis_length**2 - minor_axis_length**2)/major_axis_length
print(eccentricity)
cv2.imshow('Digital Image2', border)
cv2.waitKey(0)
cv2.destroyAllWindows()
