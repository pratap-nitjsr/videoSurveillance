import numpy as np
import cv2
import os
from matplotlib import pyplot as plt



images_names = os.listdir('D:/Milan/Desktop/projects/CapStoneProject/images')



images = np.array([cv2.imread(f'D:/Milan/Desktop/projects/CapStoneProject/images/{img}') for img in images_names ])
# images = [np.array([
#   [[142, 85, 230], [200, 15, 123], [50, 255, 90]], 
#   [[77, 180, 240], [220, 50, 10], [100, 75, 255]],   
#   [[180, 200, 30], [15, 90, 140], [250, 60, 180]]   
# ])]




def bits_to_dna_matrix(b1, b2, rule):
    
    if rule == 1 :
        if b1 == 0 and b2 == 0:
            return 'A'
        elif b1 == 0 and b2 == 1:
            return 'C'
        elif b1 == 1 and b2 == 0:
            return 'G'
        elif b1 == 1 and b2 == 1:
            return 'U'
        
    elif rule == 2 :
        if b1 == 0 and b2 == 0:
            return 'A'
        elif b1 == 0 and b2 == 1:
            return 'G'
        elif b1 == 1 and b2 == 0:
            return 'C'
        elif b1 == 1 and b2 == 1:
            return 'U'
        
    elif rule == 3 :
        if b1 == 0 and b2 == 0:
            return 'C'
        elif b1 == 0 and b2 == 1:
            return 'A'
        elif b1 == 1 and b2 == 0:
            return 'U'
        elif b1 == 1 and b2 == 1:
            return 'G'
        
    elif rule == 4 :
        if b1 == 0 and b2 == 0:
            return 'G'
        elif b1 == 0 and b2 == 1:
            return 'A'
        elif b1 == 1 and b2 == 0:
            return 'U'
        elif b1 == 1 and b2 == 1:
            return 'C'
        
    elif rule == 5 :
        if b1 == 0 and b2 == 0:
            return 'C'
        elif b1 == 0 and b2 == 1:
            return 'U'
        elif b1 == 1 and b2 == 0:
            return 'A'
        elif b1 == 1 and b2 == 1:
            return 'G'
        
    elif rule == 6 :
        if b1 == 0 and b2 == 0:
            return 'G'
        elif b1 == 0 and b2 == 1:
            return 'U'
        elif b1 == 1 and b2 == 0:
            return 'A'
        elif b1 == 1 and b2 == 1:
            return 'C'
        
    elif rule == 7 :
        if b1 == 0 and b2 == 0:
            return 'U'
        elif b1 == 0 and b2 == 1:
            return 'C'
        elif b1 == 1 and b2 == 0:
            return 'G'
        elif b1 == 1 and b2 == 1:
            return 'A'
        
    elif rule == 8 :
        if b1 == 0 and b2 == 0:
            return 'U'
        elif b1 == 0 and b2 == 1:
            return 'G'
        elif b1 == 1 and b2 == 0:
            return 'C'
        elif b1 == 1 and b2 == 1:
            return 'A'
        
    else :
        print("Enter valid rule number.")


def split_image_into_bit_planes(image):
    rows, columns = image.shape
    bit_plane = np.array([np.zeros((rows,columns), dtype=np.uint8) for _ in range(8)])
    for i in range(8):
        bit_plane[i] = (image>>i) & 1
        bit_plane[i]*=255
    return bit_plane


def Encoding_images(images, rule):
    for i,image in enumerate(images):
        bit_planes = []
        r_channel, g_channel, b_channel = image[:,:,2] , image[:,:,1],image[:,:,0]
        channel_list = [r_channel, g_channel, b_channel]
        for channel in channel_list:
            bit_planes.append(split_image_into_bit_planes(channel))
#         make_DNA_matrices(bit_planes, rule)
        save_bit_planes(bit_planes, f"output{i}")
        

def make_DNA_matrices(bit_planes, rule):
    for channel in bit_planes:
        rows, columns = channel.shape[1:]
        matrix = np.array([np.empty((rows,columns), dtype='U6') for _ in range(4)])
        for x in range(rows):
            for y in range(columns):
                for j in range(4):
                    j1 = 2 * j + 1
                    j2 = 2 * j 
                    matrix[j][x, y] = bits_to_dna_matrix(channel[j1][x][y], channel[j2][x][y], rule)
        print(matrix)
        
def save_bit_planes(bit_planes, output_prefix):
    for i, bit_plane in enumerate(bit_planes[0]):
        print(cv2.imwrite(f"{output_prefix}_bit_plane_{7-i}.jpg", bit_plane))




Encoding_images(images,7)

# images stored at :  C:\Users\milan\CapStone_Project




