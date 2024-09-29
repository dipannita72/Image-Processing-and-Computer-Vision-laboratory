import cv2
import numpy as np
import math
from tabulate import tabulate
from scipy.spatial import distance
def calculate_points(img):

    min_x, max_x, min_y, max_y = float('inf'), 0, float('inf'), 0

    # Iterate through the border pixels
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] == 255:  # If the pixel is part of the border
                # Update minimum and maximum x and y values
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    #print(leftmost, rightmost, topmost, bottommost)
    a = max_x - min_x#np.sqrt((rightmost[0] - leftmost[0]) ** 2 + (rightmost[1] - leftmost[1]) ** 2)
    b = max_y-min_y#np.sqrt((topmost[0] - bottommost[0]) ** 2 + (topmost[1] - bottommost[1]) ** 2)
    diameter = max(a,b)
    #print(diameter)
    return diameter

def calculate_descriptors(image):
    #_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    border = image - eroded
    cv2.imshow('Digital Image2', border)


    perimeter = np.count_nonzero(border)
    area = np.count_nonzero(image)
    diameter = calculate_points(border)
    print(perimeter, area,diameter)

    #calculate descriptors
    form_factor = (4*np.pi*area)/(perimeter**2)
    roundness = (4*area)/(np.pi*diameter**2)
    compactness = (perimeter**2)/ area
    #print(form_factor,roundness,compactness)
    return form_factor,roundness,compactness
    
'''image = 255*np.array((
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ), dtype="uint8")'''
    
descriptors=[]
#descriptors.append([' ', '    form_factor','      roundness','            compactness'])
image_name = ['c1.jpg','t1.jpg','ss.jpg','c2.jpg','t2.jpg','p2.png','st.jpg']
im = ['c1','t1','s1','c2','t2','s2','st']
for i in range(len(image_name)):
    img = cv2.imread(image_name[i], 0)
    f,r,c = calculate_descriptors(img)
    descriptors.append([f,r,c])
#print(descriptors)

# Define the file path
file_path = 'output.txt'

# Open the file for writing
with open(file_path, 'w') as file:
    # Write the column names
    file.write('\t'.join(map(str, [' ', '    form_factor','      roundness','            compactness'])) + '\n')

    # Write horizontal line
    file.write('-' * (15* len(descriptors[0])) + '\n')
    i=0
    # Write the data rows
    for row in descriptors[0:]:
        #print(im[i],row)
        # Convert the elements to strings and join them with tabs
        line = im[i]
        line = '\t'.join(map(str,  row))
        i=i+1
        # Write the line to the file
        file.write(line + '\n')
        # Write horizontal line
        file.write('-' * (15 * len(descriptors[0])) + '\n')
        
distances_matrix = []
for i in range(0,len(descriptors)):
    distances_row = []
    for j in range(0, len(descriptors)):
        if i == j:
            distances_row.append(0)  # Distance from a row to itself is 0
        else:
            sum_squared_diff = sum((a - b) ** 2 for a, b in zip(descriptors[i], descriptors[j]))
            dist = math.sqrt(sum_squared_diff)
            #dist = distance.euclidean(descriptors[i], descriptors[j])
            formatted_dist = "{:.2f}".format(dist)
            distances_row.append(formatted_dist)
    distances_matrix.append(distances_row)
#print(distances_matrix)
    
row_headers = [f'Test {i + 1}' for i in range(4)]
col_headers = [f'GT {i + 1}' for i in range(3)]

distances_matrix = np.array(distances_matrix)
# Display the distance matrix as a table
print(tabulate(distances_matrix[3:7,0:3], headers=col_headers, showindex=row_headers, tablefmt='grid'))

cv2.waitKey(0)
cv2.destroyAllWindows()