#!/usr/bin/env python3

"""
@author: pvltarife
"""
#Dataset and experiment
iterations = 70
mapList=['PAC_0']
testProp=0.0
numFolds=5

#General parameters
# imSize=(141,141)
imSize=(224,224)
modelType='Resnet18'
polar_coords=True

#Training parameters
lr=1e-2 #Learning Rate
momentum=0.9 #Momentum
wd=1e-5 #Weight Decay
step_size=2 
gamma=0.5
train_bs = 64
train_num_workers = 3
num_epochs = 50
max_epochs_no_improvement = 5
dropout=0.0
angle_range_da = 15.0
jitter_brightness = 0.01
jitter_contrast = 0.00

#Data augmentation
centerMethod = 0 #0 no center, 1 mask, 2 minDensity
cropEyeBorder = 1

#Validation parameters
val_bs = 64
val_num_workers = 3
