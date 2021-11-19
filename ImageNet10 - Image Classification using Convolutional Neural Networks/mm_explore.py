# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:28:17 2021

@author: shilpa
"""

import argparse
import torch
from PIL import Image
import numpy as np
import torchvision
from torch import nn, optim
from torchvision import  datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import scipy.misc

# Set up training arguments and parse
parser = argparse.ArgumentParser(description='Explore pre-trained AlexNet')

parser.add_argument(
     '--image_path', type=str,
     help='Full path to the input image to load.')
parser.add_argument(
     '--use_pre_trained', type=bool, default=True,
     help='Load pre-trained weights?')

args = parser.parse_args()

# Device configuration - defaults to CPU unless GPU is available on device
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# print("=======================================")
# print("                PARAMS               ")
# print("=======================================")

for arg in vars(args):
     print(F"{arg:>20} {getattr(args, arg)}")

#########################################################################
#
#        QUESTION 2.1.2 code here - Read Image & transform
# 
#########################################################################

device="cpu"

# Read image from path    
img = Image.open(args.image_path)

# convert image to rgb
try:
  img = img.convert('RGB') # To deal with some grayscale images in the data
except:
  pass

# apply transformation ie convert to tensor & normalize
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
        #transforms.Resize(128),
        #transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
img = transform(img) #torch.Size([3, 355, 334])

# change the dimension as per alexnet shape requirement
img = img.unsqueeze(0)#torch.Size([1, 3, 355, 334])

# Load the alexnet model from github
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
#set the model evalauation phase to true 
model.eval()

# Pass image through a single forward pass of the network
# function that does the forward pass of the image on alexnet model    
def evaluate_alexnet(model):
    with torch.no_grad():
        #images = images.to(device)
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        print("predicted: ",predicted)
    return "Done"

# invoke the alexnet model prediction function
result = evaluate_alexnet( model)
print(result)


#########################################################################
#
#        QUESTION 2.1.3 - Extract filters
# 
#########################################################################
# layer indices of each conv layer in AlexNet
conv_layer_indices = [0, 3, 6, 8, 10]

def extract_filter(conv_layer_idx, model):
        # Extracts a single filter from the specified convolutional layer,
		# zero-indexed where 0 indicates the first conv layer.
 
		# Args:
			# conv_layer_idx (int): index of convolutional layer
			# model (nn.Module): PyTorch model to extract from
	# Extract filter
    the_filter = model.features[conv_layer_idx].weight.data
    return the_filter
    
for conv_index in conv_layer_indices:
    filter_weights = extract_filter(conv_index,model)
    print("Filter at {}, is {}".format(conv_index, filter_weights))
    

#########################################################################
#
#        QUESTION 2.1.4 - Extract feature maps
# 
#########################################################################

# Function that extracts all feature map at relu layer and adds to a list and returns the list
def extract_feature_maps(input, model):
	# Args:
		# input (Tensor): input to model
		# model (nn.Module): PyTorch model to extract from
	# Extract all feature maps
	# Hint: use conv_layer_indices to access 
    img = input
    print("img shape:",img.shape)
    relu_feature_index_for_featuremap_list = [1,4,7,9,11]
    feature_map_list = []
    features = model.features
    with torch.no_grad():
        for index,layer in enumerate(features):
            # pass the image to the layer and retreive the feature map, the output image ie featuremap is passed as input to next layer
            img = layer(img)
            if index in relu_feature_index_for_featuremap_list:
                feature_map_list.append(img)
    return feature_map_list

all_relu_feature_maps_list = extract_feature_maps(img, model)
print("all_relu_feature_maps_list--",all_relu_feature_maps_list)

#########################################################################
#
#        QUESTION 2.2 - Plot Filter at layer
# 
#########################################################################

# Method plots the filter of the passed conv index for the 0th channel
def show_filter(weight_of_filter_at_index):
    plt.figure(figsize=(20, 17))
    # for k in range(len(model_weights)):
    for i, filter in enumerate(weight_of_filter_at_index):
        plt.subplot(8, 8, i+1) 
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
    plt.show()
    
at_layer_index = 0    
weight_of_filter_at_index =extract_filter( at_layer_index,model)
show_filter(weight_of_filter_at_index)

#########################################################################
#
#        QUESTION 2.2 - Plot Feature Map at layer
# 
#########################################################################

# Method extracts single feature map at relu layer and returns the single feature map
def extract_single_feature_maps(input, model, get_feature_at):
	# Args:
		# input (Tensor): input to model
		# model (nn.Module): PyTorch model to extract from
	# Extract all feature maps
	# Hint: use conv_layer_indices to access 
    img = input
    print("img shape:",img.shape)
    features = model.features
    with torch.no_grad():
        for index,layer in enumerate(features):
            img = layer(img)
            if get_feature_at == index:
                return img

# index of relu feature map are 1,4,7,9,11
at_layer_index = 1
single_feature_map = extract_single_feature_maps(img, model, at_layer_index)
#print("single_feature_map:", single_feature_map)

# Method plots the featuremap of the passed feature map tensor 
def show_feature_map(feature_map):
    print(feature_map.shape)
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure(figsize=(20, 17))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.subplot(8, 8, index)
        plt.imshow(feature_map[index-1], cmap='gray')
        plt.axis('off')
        if(index==64):
            break
    plt.show()
    
show_feature_map(single_feature_map)    

