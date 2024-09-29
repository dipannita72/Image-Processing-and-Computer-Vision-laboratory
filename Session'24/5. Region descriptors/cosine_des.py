import cv2
import numpy as np
import math
from tabulate import tabulate
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

def calculate_points(img):
    
    min_x, max_x, min_y, max_y = float('inf'), 0, float('inf'), 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] == 255:  # If the pixel is part of the border
                # Update minimum and maximum x and y values
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    #print(leftmost, rightmost, topmost, bottommost)
    a = max_x - min_x
    b = max_y-min_y
    diameter = max(a,b)
    return diameter

def calculate_descriptors(image):
    #_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    border = image - eroded
    #cv2.imshow('Digital Image2', border)


    perimeter = np.count_nonzero(border)
    area = np.count_nonzero(image)
    diameter = calculate_points(border)
    #print(perimeter, area)

    #calculate descriptors
    form_factor =round( (4*np.pi*area)/(perimeter**2),2)
    roundness = round((4*area)/(np.pi*diameter**2),2)
    compactness =round( ((perimeter**2)/ area),2)
    #print(form_factor,roundness,compactness)
    
    return form_factor,roundness,compactness

    
descriptors=[]
image_name = ['c1.jpg','t1.jpg','p1.png','c2.jpg','t2.jpg','p2.png','st.jpg']
im = ['c1','t1','s1','c2','t2','s2','st']
for i in range(len(image_name)):
    img = cv2.imread(image_name[i], 0)
    f,r,c = calculate_descriptors(img)
    descriptors.append([f,r,c])
#print(descriptors)

file_path = 'output.txt'
with open(file_path, 'w') as file:
    file.write('\t'.join(map(str, [' ', '    form_factor','      roundness','            compactness'])) + '\n')
    file.write('-' * (15* len(descriptors[0])) + '\n')
    i=0
    for row in descriptors[0:]:
        line = im[i]
        line = '\t'.join(map(str,  row))
        i=i+1
        file.write(line + '\n')
        file.write('-' * (15 * len(descriptors[0])) + '\n')
        
distances_matrix = []
for i in range(0,len(descriptors)):
    distances_row = []
    for j in range(0, len(descriptors)):
            dot_product = np.dot(descriptors[i], descriptors[j])
            magnitude_vector1 = np.sqrt(np.sum(np.array(descriptors[i])**2))
            magnitude_vector2 = np.sqrt(np.sum(np.array(descriptors[j])**2))
            cosine_sim = dot_product / (magnitude_vector1 * magnitude_vector2)
            
            matrix1 = descriptors[i] / magnitude_vector1 #np.array(descriptors[i]).reshape(1, -1)
            matrix2 = descriptors[j] / magnitude_vector2 #np.array(descriptors[j]).reshape(1, -1)
            cos = np.dot(matrix1, matrix2)
            #formatted_dist = "{:.4f}".format(cosine_similarity)
            distances_row.append((cos))
    distances_matrix.append(distances_row)
#print(distances_matrix)
    
row_headers = [f'Test {i + 1}' for i in range(4)]
col_headers = [f'GT {i + 1}' for i in range(3)]

distances_matrix = np.array(distances_matrix)
# Display the distance matrix as a table
print(tabulate(distances_matrix[3:7,0:3], headers=col_headers, showindex=row_headers, tablefmt='grid'))
#[3:7,0:3]
cv2.waitKey(0)
cv2.destroyAllWindows()