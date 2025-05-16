#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:32:36 2023

@author: igonzalez
"""

#Dataset and experiment
db_path = "../datasets"
csvFile='../datasets/dataset_global_unet/patient_list.csv'
imageDir='images'
dataDir = 'data'
error_PACDir = 'error_PAC'
error_BFSDir = 'error_BFS'

mapList=['PAC_0']

random_seed = 0
dataset_seed = 0
testProp=0.0
numFolds=5
iterations = 70

#General parameters
imSize=(224,224)
modelType= 'Resnet18'
polar_coords=True
patch_hiding=True

#Training parameters
lr=1e-2
momentum=0.9
wd=1e-5
step_size=2
gamma=0.5
train_bs = 32
train_num_workers = 2
num_epochs = 50
max_epochs_no_improvement = 10
dropout=0.0
angle_range_da = 15.0
jitter_brightness = 0.01
jitter_contrast = 0.00

#Data augmentation
centerMethod = 0 #0 no center, 1 mask, 2 minDensity
cropEyeBorder = 1

#Validation parameters
val_bs = 32
val_num_workers = 2
grad_cam = True
