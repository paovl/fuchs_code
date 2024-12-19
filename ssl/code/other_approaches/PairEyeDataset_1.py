#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:34:14 2024

@author: mariagonzalezgarcia
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
from dataAugmentation import Normalize, ToTensor
from skimage.transform import resize
from sklearn.preprocessing import label_binarize
import ast
from itertools import combinations
import matplotlib.pyplot as plt

class PairEyeDataset(Dataset):
    def __init__(self, csv_file, image_dir, data_dir, transform=None, test=False, random_seed=42, testProp=0.0, mapList=None, imSize=(141,141)):
        self.image_dir = image_dir
        self.data_dir = data_dir
        self.dataset = pd.read_csv(csv_file, header=0, dtype={'Sesion': str})
        self.ojo_str = ["OD","OS"]
        self.imSize = imSize
        self.transform = transform
        
        if mapList is None:
            self.mapList = ['ELE_0', 'ELE_1', 'CUR_0', 'CUR_1', 'CORNEA-DENS_0', 'CORNEA-DENS_1', 'PAC_0']
        else:
            self.mapList = mapList
        
        self.image_pairs = self.load_image_pairs()
        
        if testProp > 0:
            # We divide the dataset into train and test by using the corresponding random_seed
            np.random.seed(random_seed)
            total_data = len(self.dataset)
            testData = int(testProp * total_data)
            idx = np.random.permutation(range(total_data))
            if test:
                idx = idx[:testData]
            else:
                idx = idx[testData:]
            self.dataset = self.dataset.iloc[idx]
            self.dataset = self.dataset.reset_index(drop=True)
            print(idx)
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image1_path, image2_path, label = self.image_pairs[idx]
        nMaps = len(self.mapList)
        image1 = np.empty((self.imSize[0], self.imSize[1], nMaps), dtype=np.float32)
        image2 = np.empty((self.imSize[0], self.imSize[1], nMaps), dtype=np.float32)
        
        for i in range(nMaps):
            split_map = self.mapList[i].split('_')
            split_img_path1 = image1_path.split('_')
            split_img_path2 = image2_path.split('_')
            impath1 = split_img_path1[0] + '_' + split_map[0] + '_' + split_img_path1[1] + '_' + split_img_path1[2] + '_' + split_map[1] + '.npy'
            cmap1 = np.load(self.data_dir + '/' + impath1)
            impath2 = split_img_path2[0] + '_' + split_map[0] + '_' + split_img_path2[1] + '_' + split_img_path2[2] + '_' + split_map[1] + '.npy'
            cmap2 = np.load(self.data_dir + '/' + impath2)
            (h, w) = cmap1.shape
            if w != self.imSize[0] or h != self.imSize[1]:
                cmap1 = resize(cmap1, self.imSize, preserve_range=True)
                cmap2 = resize(cmap2, self.imSize, preserve_range=True)
            image1[:, :, i] = cmap1
            image2[:, :, i] = cmap2
        
        if self.transform:
            image1, image2, label = self.transform((image1,image2,label))
            
        return image1, image2, label
    
    def load_image_pairs(self):
        image_pairs = []
        self.dataset["Row"] = range(0, len(self.dataset))
        
        # Load similar image pairs
        for (idx, ojo), group in self.dataset.groupby(['Index', 'Ojo']):
            sesiones = list(group['Temp'])
            filas = list(group['Row'])
            if len(sesiones) > 1:
                for a, b in combinations(filas, 2):
                    # Ensure the sessions are different
                    if self.dataset.loc[self.dataset['Row'] == a, 'Temp'].values[0] != self.dataset.loc[self.dataset['Row'] == b, 'Temp'].values[0]:
                        imgpath_1 = str(idx) + '_' + ojo + '_' + self.dataset.loc[a]["Sesion"]
                        imgpath_2 = str(idx) + '_' + ojo + '_' + self.dataset.loc[b]["Sesion"]
                        image_pairs.append((imgpath_1, imgpath_2, 0)) 
        
        # Load dissimilar image pairs
        for name, group in self.dataset.groupby(['Ojo']):
            todas_combinaciones = combinations(group['Row'], 2)
            for a, b in todas_combinaciones:
                imgpath_1 = str(self.dataset.loc[a]['Index']) + '_' + name[0] + '_' + self.dataset.loc[a]["Sesion"]
                imgpath_2 = str(self.dataset.loc[b]['Index']) + '_' + name[0] + '_' + self.dataset.loc[b]["Sesion"]
                if (imgpath_1, imgpath_2, 0) in image_pairs:
                    continue
                else:
                    image_pairs.append((imgpath_1, imgpath_2, 1))
        
        return image_pairs


    
if __name__ == '__main__':
    csvFile = '../dataset/patient_list.csv'
    imageDir = '../dataset/images'
    dataDir = '../dataset/data'
    mapList = ['ELE_0', 'ELE_1', 'CUR_0', 'CUR_1', 'CORNEA-DENS_0', 'CORNEA-DENS_1', 'PAC_0']
    
    normalization_file = os.path.join(dataDir, 'normalization.npz')
    with open(normalization_file, 'rb') as f:
        normalization_data = pickle.load(f)
    
    # Filter using the map list
    idxCat = [normalization_data['categories'].index(cat) for cat in mapList]
    
    transform_chain = transforms.Compose([ToTensor(),
                                          Normalize(mean=normalization_data['mean'][idxCat],
                                          std=normalization_data['std'][idxCat])])
    
    dataset = PairEyeDataset(csvFile, imageDir, dataDir, transform=transform_chain, mapList=mapList, test=False, random_seed=42, testProp=0.2)
    
    #dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    #for i, sample in enumerate(dataloader):
       # print(f'Sample {i} processed.\n')
        #if i == 2:  # Limitar el número de muestras para no sobrecargar la salida
           # break
