#import libraries

import re
import cv2
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance


RANDOM_SEED = 58



def list_directory_contents(directory):

    pictures_list = []

    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                #print(entry.name)
                pictures_list.append(entry.name)
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
    except NotADirectoryError:
        print(f"{directory} is not a directory.")
    except PermissionError:
        print(f"Permission denied to access {directory}.")
    return pictures_list

class RandomZoomAndShift:
    def __init__(self, zoom_range=(0.8, 1.2),shift_h_range = 0,shift_v_range = None):
        self.zoom_range = zoom_range
        self.shift_h_range = shift_h_range
        self.shift_v_range = shift_v_range #we are not gonna use theses params in fact, or maybe later. For the moment, we will set h and v shift with the constraist that at max zoom, we cannot move without putting a point out of bounds

    def __call__(self, img):
        zoom_factor = random.uniform(*self.zoom_range)
        width, height,_ = img.shape
        h_range = int((self.zoom_range[1]-zoom_factor)/2*width)
        self.shift_h_range = zoom_factor
        v_range = int((self.zoom_range[1]-zoom_factor)/2*width)
        pos_factor = int(random.uniform(-h_range,h_range)),int(random.uniform(-v_range,v_range))
        self.shift_v_range = pos_factor
        new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
        img = cv2.resize(img, (new_height, new_width), interpolation=cv2.INTER_LINEAR)

        #print(zoom_factor)

        padding_value = 200
        padding =  ((padding_value, padding_value), (padding_value, padding_value), (0, 0))
        img = np.pad(img, padding, mode='constant', constant_values=0)
        # Calculate cropping coordinates to keep the original size
        left = max(0,(new_width-width)//2 + pos_factor[0]+padding_value)
        top = max(0,(new_height -height)//2 + pos_factor[1]+padding_value)
        right = left + width
        bottom = top + height
        #print(pos_factor)
        #image_augmentation


        img = img[left:right, top:bottom]
        #print(img.shape)
        if img.shape != (256,512,3):
            print(img.shape)
            print(zoom_factor)
        return img
    
# class RandomRotation: see you later randomrotation, I will code you later in the future you understand that

        
class RandomDownQuality:
    def __init__(self,quality_delta = 0.5): #we set this parameter given that worst quality images from adrien's dataset are with 100*200 shape
        self.quality_delta = quality_delta
    
    def __call__(self, img):
        down_quality_factor = min(random.uniform(1-self.quality_delta,1+self.quality_delta),1) #we want that most of images are not down qualitized, but upqualitizing not possible of course
        ini_width,ini_height,_ = img.shape
        new_width,new_height = int(img.shape[0] * down_quality_factor),int(img.shape[1] * down_quality_factor)
        img = cv2.resize(img, (new_height, new_width), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(img, (ini_height,ini_width),interpolation=cv2.INTER_LINEAR)
        return img
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,padding=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),   #we put padding = 0 in order to avoid border effect with border detection
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module): #I reduce the number of filters in the first layer to 16 in order to avoid overfitting
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 16)  
        self.encoder2 = DoubleConv(16, 32)
        self.encoder3 = DoubleConv(32, 64)
        self.encoder4 = DoubleConv(64, 128)
        self.encoder5 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(256, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(128, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(64, 32)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(32, 16)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.final_activation = nn.Hardsigmoid()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3)) 
        enc5 = self.encoder5(self.pool(enc4)) 

        
        dec4 = self.upconv4(enc5) 
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_activation(self.final_conv(dec1))
    




def exponential_relief(distance_image,l):
    """this function take a distances image as parameter and return np.exp(-lambda * distance_image)"""
    return np.exp(-l *  distance_image)


def single_point_distance_transform(image_shape, point):
    # Create a grid of coordinates
    y, x = np.indices(image_shape)

    # Calculate the distance from the point to all other points
    distance = np.sqrt(np.square(x - point[1]) + np.square(y - point[0]))

    return distance





def second_closest_distance_transform(points):
    # Get coordinates of non-zero pixels
    output_size = (256,512)
    binary_image = np.zeros(output_size,dtype = np.uint8)
    for pt in points:
        binary_image[pt[0],pt[1]] = 1

    # Initialize the distance transform array
    second_closest_distances = np.full(binary_image.shape, np.inf)


    # Compute distances for each pixel
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
                # Compute distances to all non-zero pixels
                distances = distance.cdist([(i, j)], points, 'euclidean').flatten()

                if len(distances) > 1:
                    # Sort distances and take the second smallest
                    sorted_distances = np.sort(distances)
                    second_closest_distances[i, j] = sorted_distances[1]


    return second_closest_distances





def exponential_relief_inverse(expo_image,l):
    return np.exp(1/l * expo_image)

def creation_relief_v3(points,relief_function,parameters):
    """relief function must be a vectorial function which is decreasing and have values from 1 to 0 on R+
    this function take points as argument and return the density map corresponding to theses points
    parameters must be a dict with relief_function additional parameters"""
    output_size = (256,512)
    binary_image = np.ones(output_size,dtype = np.uint8)
    for pt in points:
        binary_image[pt[0],pt[1]] = 0
    dist_trans = cv2.distanceTransform(binary_image, cv2.DIST_L2, 0)
    return relief_function(dist_trans,**parameters)

def creation_relief_ulti_v2(points : np.ndarray,r : list,relief_functions : dict,parameters : dict) -> np.ndarray:
    """different implementation as below : we first conpute one relief by point, then take the maximum for each point of the mapping"""
    output_size = (256,512)

    mapping = np.zeros(output_size)

    for index,pt in enumerate(points):

        current_map = single_point_distance_transform(output_size,pt)

        current_map_prepared =  (current_map < r[index]+1).astype(np.uint8)*(r[index] -current_map)/r[index]

        current_function = relief_functions[index]

        current_parameters = parameters[index]

        mapping = np.maximum(mapping,current_function(current_map_prepared,*current_parameters))

    return mapping


def creation_relief_ulti_SingleLayer(points : np.ndarray,r : list,relief_functions : dict,parameters : dict) -> np.ndarray:
    """this is the ultimate version of creation_relief, that allow to create any relief you want, with specific treatment for each point if needed
    the control of r, the parameters for each function in the dictionnary relief_functions
    parameters :
    -points : input points to create the density map
    -r  : radius
    -relief_functions : dict, ex : {0 : function_1, 1 : function_2, ...}
    note : relief_function_i must be a function from [0,1] to [0,1], vectorised, so it can operate on arrays
    -parameters : dict(dict), ex : {0 : params_1, 1 : params_2, ...} where params_i if the dictionnary with parameters for corresponding function_i
    """
    output_size = (256,512) #define output size

    binary_image = np.ones(output_size,dtype = np.uint8) #defining binary image to perform distance transform
    for pt in points:
        binary_image[pt[0],pt[1]] = 0

    dist_trans,labels = cv2.distanceTransformWithLabels(binary_image, cv2.DIST_L2, 0,labelType=cv2.DIST_LABEL_PIXEL) #performing distance transform

    correspondence = dict(zip([labels[pt[0],pt[1]] for pt in points],[i for i in range(1,len(points)+1)])) #correspondence beetween distance_transform labels and True labels

    vectorized_correspondence = np.vectorize(correspondence.get) #we transform the correspondence dictionnary into a vectorised function

    true_labels = vectorized_correspondence(labels) #changing labels to fit initial order

    true_dist_trans = 1-dist_trans/np.max(dist_trans)

    final_mapping = np.zeros(output_size)

    for index in range(len(points)):
        
        current_zone_mask = ((dist_trans < r[index]).astype(np.uint8)*(true_labels == index+1).astype(np.uint8)).astype(bool)

        current_function = relief_functions[index]

        current_parameters = parameters[index]

        final_mapping[current_zone_mask] = current_function(true_dist_trans[current_zone_mask],*current_parameters)

def creation_relief_v3_full(points,relief_function,parameters):
    heat_map_layer1 = creation_relief_v3(points,relief_function,parameters)
    heat_map_layer1 = np.expand_dims(heat_map_layer1,axis = 2)
    heat_map_layer2 = exponential_relief(second_closest_distance_transform(points),**parameters)
    heat_map_layer2 = np.expand_dims(heat_map_layer2,axis = 2)
    heat_map_full = np.concatenate((heat_map_layer1,heat_map_layer2),axis = 2)
    return heat_map_full
