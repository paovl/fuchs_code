#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:39:06 2023

@author: igonzalez
"""

from PIL import Image
# import _init_paths
import os
import torch
import pandas as pd
from skimage import io, transform, color, morphology, util
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import numpy as np
import cv2
import pdb
import pickle
from dataAugmentation_unet import Normalize, ToTensor
from skimage.transform import resize
import matplotlib.pyplot as plt

class EyeDataset(Dataset):
  """Eye dataset."""

  def __init__(self, csv_file, image_dir,transform=None,test=False,random_seed=42,testProp=0.0,mapList=None,imSize=(141,141)):
      """
      Args:
          csv_file (string): Path al fichero csv con las anotaciones.
          image_dir (string): Directorio raíz donde encontraremos las carpetas 'images' y 'masks' .
          transform (callable, optional): Transformaciones opcionales a realizar sobre las imágenes.
          testProp (int): Proporción de datos que se utilizan para test
      Var:
          dataset: dict con 'id' (path y nombre de las imágenes), 'label' (bbox) y 'segs' (matrices de segmentación)
      """
      self.image_dir=image_dir
      self.dataset = pd.read_csv(csv_file,header=0,dtype={'img_id': str, 'label': int}) #dictionary variable
      self.ojo_str=["OD","OS"]
      self.imSize=imSize
      self.transform = transform
      
      if mapList is None:
          self.mapList=['ELE_0', 'ELE_1', 'CUR_0','CUR_1', 'CORNEA-DENS_0','CORNEA-DENS_1','PAC_0']
      else:
          self.mapList=mapList
      if testProp>0:
          
          #We divide the dataset into train and test by using the corresponding random_seed
          np.random.seed(random_seed)

          # Choose the dataset
          total_data=len(self.dataset)
          testData=int(testProp*total_data)
          idx=np.random.permutation(range(total_data)) #Create a random index array  (0-total_data)
          #Chooses the index of images that belong to the train or test dataset
          if test:
              idx=idx[:testData]
          else:
              idx=idx[testData:]
          self.dataset=self.dataset.iloc[idx]
          self.dataset=self.dataset.reset_index(drop=True)  
          print(idx)
              
  def __len__(self):
      return len(self.dataset)

  def __getitem__(self, idx): # carga imagen y su mask, seg y labels y lo recoge todo en sample (que es un dict)  
      nMaps=len(self.mapList)
      #Read
      image=np.empty((self.imSize[0],self.imSize[1],nMaps),dtype=np.float32)

      for i in range(nMaps):
          impath='%s_%s.npy'%(self.dataset.iloc[idx]['img_path'],self.mapList[i]) #path to the numpy data 
          cmap=np.load(impath)
          # cmap=cmap/255
          (h,w)=cmap.shape
          if w!=self.imSize[0] or h!=self.imSize[1]:
              cmap=resize(cmap,self.imSize,preserve_range=True)  
          image[:,:,i] = cmap
      
      # Define the variable sample 
      sample = {'image': image}
      if self.transform:
           image= self.transform(sample)
    
      img_info = {'idx': idx, 'img_path':self.dataset.iloc[idx]['img_path'], 'img_id':self.dataset.iloc[idx]['img_id'], 'img':image['image'], 'label': image['label'], 'mask': image['mask']}

      #Returns all the topographic maps of the pacient 
      return img_info
 
  def getbiomarkers(self,indices=None):
    """
        Return the biomarkers and the label for the idx indicated 
    """
     
    if indices is None:
        img_ids = self.dataset['img_id'].to_numpy
        X=self.dataset[['PaqRel','DensAnt']].to_numpy()
        Y=self.dataset['label'].to_numpy()
    else:
        dataset=self.dataset.iloc[indices]
        img_ids = dataset['img_id'].values
        X=dataset[['PaqRel','DensAnt']].to_numpy()
        Y=dataset['label'].to_numpy()
    return X, Y, img_ids
           
if __name__ == '__main__':
    """
        This is a test code 
    """
    csvFile='./dataset/annotation.csv'
    imageDir='./dataset/images'
    mapList=['ELE_0', 'ELE_1', 'CUR_0','CUR_1', 'CORNEA-DENS_0','CORNEA-DENS_1','PAC_0']
    
    # Load normalization data
    normalization_file=os.path.join(imageDir,'normalization.npz')
    with open(normalization_file, 'rb') as f:
        normalization_data = pickle.load(f)
    
    # Filter normalization data using the map list
    idxCat = [normalization_data['categories'].index(cat) for cat in mapList]
    
    
    transform_chain=transforms.Compose([ToTensor(),
                                        Normalize(mean=normalization_data['mean'][idxCat],
                                        std=normalization_data['std'][idxCat])])
    
    # Create the dataset 
    dataset=EyeDataset(csvFile, imageDir,transform=transform_chain,mapList=mapList,test=False,random_seed=42,testProp=0.2)
    