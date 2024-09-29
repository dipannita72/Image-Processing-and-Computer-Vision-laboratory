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
    a = np.sqrt((rightmost[0] - leftmost[0]) ** 2 + (rightmost[1] - leftmost[1]) ** 2)

    # Calculate minor axis length
    b = np.sqrt((topmost[0] - bottommost[0]) ** 2 + (topmost[1] - bottommost[1]) ** 2)
    major_axis_length = max(a,b)/2
    minor_axis_length = min(a,b)/2
    return major_axis_length, minor_axis_length


def find_objects_from_border(border_image):
    # Connected Components Labeling
    num_labels, labels = cv2.connectedComponents(border_image)
    
    # Get the unique labels excluding background (label 0)
    unique_labels = np.unique(labels)[1:]
    
    # Create a blank image to draw objects
    objects_image = np.zeros_like(border_image)
    i=0
    eccentricities = []
    # Find objects and draw them
    for label in unique_labels:
        object_mask = np.uint8(labels == label) * 255
        cv2.imshow('sddd'+str(i), object_mask)
        leftmost, rightmost, topmost, bottommost = find_border_extreme_points(object_mask)
        if leftmost is not None and rightmost is not None and topmost is not None and bottommost is not None:
            # Calculate major and minor axis lengths
            major_axis_length, minor_axis_length = calculate_axes_lengths(leftmost, rightmost, topmost, bottommost)
            print(major_axis_length,minor_axis_length)
            # Calculate eccentricity as the ratio of major and minor axes lengths
            eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
            eccentricities.append(eccentricity)
            
        objects_image = cv2.bitwise_or(objects_image, object_mask)
        i=i+1
    return objects_image,eccentricities


def calculate_(objects_image,border_image):
    num_labels, labels = cv2.connectedComponents(objects_image)
    unique_labels = np.unique(labels)[1:]
    _, borderlabels = cv2.connectedComponents(border_image)
    unique_borderlabels = np.unique(borderlabels)[1:]
    i=0
    compactnesses = []
    circularitys = []
    eccentricities = []
    for label in unique_labels:
        border = unique_borderlabels[i]
        object_mask = np.uint8(labels == label) * 255
        object_mask_2 = np.uint8(borderlabels == border) * 255
        cv2.imshow('sad'+str(i), object_mask)
        cv2.imwrite('objts'+str(i)+'.jpg', object_mask)
        cv2.imshow('sddd'+str(i), object_mask_2)
        
        perimeter = np.count_nonzero(object_mask_2)
        area = np.count_nonzero(object_mask)
        compactness = area / (np.pi * (perimeter ** 2) / 4)
        compactnesses.append(compactness)
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        circularitys.append(circularity)
        leftmost, rightmost, topmost, bottommost = find_border_extreme_points(object_mask)
        if leftmost is not None and rightmost is not None and topmost is not None and bottommost is not None:
            # Calculate major and minor axis lengths
            major_axis_length, minor_axis_length = calculate_axes_lengths(leftmost, rightmost, topmost, bottommost)
            print(major_axis_length,minor_axis_length)
            # Calculate eccentricity as the ratio of major and minor axes lengths
            eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
            eccentricities.append(eccentricity)
            
        objects_image = cv2.bitwise_or(objects_image, object_mask)
        
        i = i+1
        

    return objects_image,eccentricities,compactnesses,circularitys
# Example usage:
image = cv2.imread('a1.png', cv2.IMREAD_GRAYSCALE)

# Apply binary thresholding
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(binary, kernel, iterations=1)
border = binary - eroded

# Find objects from the border
#objects_image,eccentricities = find_objects_from_border(border)
objects_image,eccentricities,compactnesses,circularitys = calculate_(binary,border)
print(compactnesses,circularitys,eccentricities)
# Display the objects found
cv2.imshow("Objects", objects_image)
cv2.waitKey(0)
cv2.destroyAllWindows()






'''
[0.05673065873845605, 0.08965513185543285, 0.038647971080370463, 0.08854528414398576, 0.057006435325302626, 0.07957747154594767]
[0.5599091591617643, 0.8848606839406268, 0.3814401854679985, 0.8739069260831895, 0.5626309649770226, 0.7853981633974483]
for label in unique_labels:
    # Create a mask for the object
    object_mask = np.uint8(labels == label)

    # Find contours of the object
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]  # Assuming there's only one contour per object
    
    # Calculate area and perimeter of the object
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate compactness as the ratio of area to the area of the minimum enclosing circle
    compactness = area / (np.pi * (perimeter ** 2) / 4)
    compactnesses.append(compactness)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    circularitys.append(circularity)
'''