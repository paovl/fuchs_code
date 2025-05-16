"""
Created on Mon Feb 10 11:49:10 2025
@author: pvltarife
"""

# imports 
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from skimage.transform import resize

class EyeDataset(Dataset):
    def __init__(self, csv_file, db_path, data_dir, transform=None, mapList=None, imSize=(141,141)):
        
        self.db_path = db_path
        self.data_dir = data_dir
        self.dataset = pd.read_csv(csv_file, header=0, dtype={'Sesion': str})
        self.ojo_str = ["OD","OS"]
        self.imSize = imSize
        self.transform = transform
        
        if mapList is None:
            self.mapList = ['ELE_0', 'ELE_1', 'CUR_0', 'CUR_1', 'CORNEA-DENS_0', 'CORNEA-DENS_1', 'PAC_0']
        else:
            self.mapList = mapList
            
    def __len__(self):
        
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        nMaps = len(self.mapList)
        
        image = np.empty((self.imSize[0], self.imSize[1], nMaps), dtype=np.float32)
        image_error = np.empty((self.imSize[0], self.imSize[1], nMaps), dtype=np.float32)

        image_info = self.dataset.iloc[idx]

        patient = image_info["Patient"]
        eye_type = image_info["Ojo"]
        session = image_info["Sesion"]
        db = image_info["Database"]
        temp_patient = image_info["Temp"]
        
        for i in range(nMaps):
            split_map = self.mapList[i].split('_')
            impath = str(patient) + '_' +  split_map[0] + '_' + eye_type + '_' + str(session) + '_' + split_map[1] + '.npy'

            cmap = np.load( self.db_path + '/' + 'dataset_' + db +  '_unet/' + self.data_dir + '/' + impath)
            cmap = resize(cmap, self.imSize, preserve_range=True)

            # cmap_error = np.load( self.db_path + '/' + 'dataset_' + db +  '_unet/' + self.data_dir + '_error' + '/' + impath)
            # cmap_error = resize(cmap_error, self.imSize, preserve_range=True)

            image[:, :, i] = cmap
            # image_error[:, :, i] = cmap_error
        
        if self.transform:
            image = self.transform({'image': image})   
            # image = self.transform('image': image, 'error': image_error})
        
        img_info = {'idx': idx, 'patient':patient, 'eye': eye_type, 'session': session, 'img':image['image'], 'label': image['label'], 'temp': temp_patient, 'mask': image['mask']}
        # img_info = {'idx': idx, 'patient':patient, 'eye': eye_type, 'session': session, 'img':image['image'], 'label': image['label'], 'temp': temp_patient, 'mask': image['mask'], 'error': image['error'], 'error_map': image_error['error_map']}
        return img_info
