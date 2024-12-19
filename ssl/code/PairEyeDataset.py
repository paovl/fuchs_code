#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:00:11 2024

@author: mariagonzalezgarcia
"""

from PIL import Image
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
    def __init__(self, csv_file, db_path, image_dir, data_dir,  error_BFS_dir, error_PAC_dir, transform=None, test=False, random_seed=42, testProp=0.0, mapList=None, imSize=(141,141)):
        self.db_path = db_path
        self.image_dir = image_dir
        self.data_dir = data_dir
        self.error_BFS_dir = error_BFS_dir
        self.error_PAC_dir = error_PAC_dir
        self.dataset = pd.read_csv(csv_file, header=0, dtype={'Sesion': str})
        self.ojo_str = ["OD","OS"]
        self.imSize = imSize
        self.transform = transform
        self.eyes_dataset = self.image_pairs()
        
        if mapList is None:
            self.mapList = ['ELE_0', 'ELE_1', 'CUR_0', 'CUR_1', 'CORNEA-DENS_0', 'CORNEA-DENS_1', 'PAC_0']
        else:
            self.mapList = mapList
                
        if testProp > 0:
            # We divide the dataset into train and test by using the corresponding random_seed
            np.random.seed(random_seed) # fija la semilla 
            total_data = len(self.eyes_dataset)
            testData = int(testProp * total_data)
            idx = np.random.permutation(range(total_data))
            if test:
                idx = idx[:testData]
            else:
                idx = idx[testData:]
            self.eyes_dataset = self.eyes_dataset.iloc[idx]
            self.eyes_dataset = self.eyes_dataset.reset_index(drop=True)
            print(idx)   
            
    def __len__(self):
        
        return len(self.eyes_dataset)
    
    def __getitem__(self, idx):
        
        nMaps = len(self.mapList)
        
        image1 = np.empty((self.imSize[0], self.imSize[1], nMaps), dtype=np.float32)
        image2 = np.empty((self.imSize[0], self.imSize[1], nMaps), dtype=np.float32)
        
        image_pair = self.eyes_dataset.iloc[idx]
        
        patient = image_pair["Patient"]
        eye_type = image_pair["Ojo"]
        session_a = image_pair["Sesion"]
        db = image_pair["Database"]
        
        sessions = self.dataset.loc[(self.dataset["Patient"] == patient) & (self.dataset["Ojo"] == eye_type) & (self.dataset["Database"] == db), "Sesion"]
        session_b = sessions[sessions != session_a].iloc[0]

        idx_patient1= self.dataset.loc[(self.dataset["Patient"] == patient) & (self.dataset["Ojo"] == eye_type) & (self.dataset["Sesion"] == session_a), 'Sesion'].index[0]
        idx_patient2 = self.dataset.loc[(self.dataset["Patient"] == patient) & (self.dataset["Ojo"] == eye_type) & (self.dataset["Sesion"] == session_b), 'Sesion'].index[0]

        temp_patient1= self.dataset.loc[(self.dataset["Patient"] == patient) & (self.dataset["Ojo"] == eye_type) & (self.dataset["Sesion"] == session_a), 'Temp'].values[0]
        temp_patient2 = self.dataset.loc[(self.dataset["Patient"] == patient) & (self.dataset["Ojo"] == eye_type) & (self.dataset["Sesion"] == session_b), 'Temp'].values[0]

        for i in range(nMaps):
            split_map = self.mapList[i].split('_')
            impath1 = str(patient) + '_' +  split_map[0] + '_' + eye_type + '_' + str(session_a) + '_' + split_map[1] + '.npy'
            impath2 =  str(patient) + '_' +  split_map[0] + '_' + eye_type + '_' + str(session_b) + '_' + split_map[1] + '.npy'
            cmap1 = np.load( self.db_path + '/' + 'dataset_' + db +  '/' + self.data_dir + '/' + impath1)
            cmap2 = np.load(self.db_path + '/' + 'dataset_' + db + '/' + self.data_dir + '/' + impath2)
            cmap1 = resize(cmap1, self.imSize, preserve_range=True)
            cmap2 = resize(cmap2, self.imSize, preserve_range=True)
            image1[:, :, i] = cmap1
            image2[:, :, i] = cmap2
        
            # read scores
            if split_map[0] == 'ELE' and split_map[1] == '4':
                error1_BFS =  np.load(self.db_path + '/'  + 'dataset_' + db + '/' + self.error_BFS_dir + '/' + impath1)
                error2_BFS =  np.load(self.db_path + '/'  + 'dataset_' + db + '/' + self.error_BFS_dir + '/' + impath2)
            
            if split_map[0] == 'PAC':
                error1_PAC =  np.load(self.db_path + '/'  + 'dataset_' + db + '/' + self.error_PAC_dir + '/' + impath1)
                error2_PAC =  np.load(self.db_path + '/'  + 'dataset_' + db + '/' + self.error_PAC_dir + '/' + impath2)

        if self.transform:
            image1, image2 = self.transform((image1,image2))
        
        img1_info = {'idx_patient': idx, 'idx': idx_patient1,'patient':patient, 'eye': eye_type, 'session': session_a, 'img':image1['image'], 'bsize_x': image1['bsize_x'], 'temp': temp_patient1, 'error_BFS': error1_BFS, 'error_PAC': error1_PAC}
        img2_info = {'idx_patient': idx, 'idx': idx_patient2,'patient':patient, 'eye': eye_type, 'session': session_b, 'img':image2['image'], 'bsize_x': image2['bsize_x'], 'temp': temp_patient2, 'error_BFS': error2_BFS, 'error_PAC': error2_PAC}

        return img1_info, img2_info
        
    def image_pairs(self):

        # the patients that have pairs are those who have a second session
        patients_pair = self.dataset[self.dataset['Temp'] == 2]['Patient'].unique()

        new_dataset = self.dataset[self.dataset['Patient'].isin(patients_pair)].copy()

        eyes = new_dataset[new_dataset["Temp"]==2].copy()

        eyes.reset_index(drop=True, inplace=True)

        eyes["Row"] = range(0, len(eyes))
        
        return eyes

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
    
    dataset = PairEyeDataset(csvFile, imageDir, dataDir, transform = transform_chain, mapList=mapList, test=False, random_seed=42, testProp=0.2)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    #for i, sample in enumerate(dataloader):
       #print(f'Sample {i} processed.\n')
