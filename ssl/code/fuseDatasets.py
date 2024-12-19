#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:11:14 2023

@author: igonzalez
"""
import numpy as np
import pandas as pd
import os
import glob
import pathlib
import PIL
from PIL import Image
import pandas as pd
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as img
import pdb
from skimage.transform import AffineTransform, warp
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_erosion, binary_dilation, disk
from skimage.filters import gaussian
import itertools
import sys
import seaborn as sns

def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])

def computeNorm(annFile, db, px = 100):
    dataset = pd.read_csv(annFile,header=0, dtype={'Sesion': str})
    #To check the different files

    imgs_path = np.array(db + '/' +'dataset_' + dataset['Database'].astype(str)+ '/data/' + dataset['Patient'].astype(str))

    listFiles=glob.glob(imgs_path[0] + '_*')
    categories = ['_'.join([os.path.basename(fichero).split('_')[1], os.path.basename(fichero).split('_')[-1][:-4]]) for fichero in listFiles]
    mom1=np.zeros(len(categories),)
    mom2=np.zeros(len(categories),)
    cont=np.zeros(len(categories),)
    
    for i in range(len(dataset)):
        ojo = dataset['Ojo'].values[i]
        sesion = dataset['Sesion'].values[i]
        img_path=imgs_path[i]
        for c in range(len(categories)):
            impath=img_path + '_' + categories[c][:-2] + '_' + ojo + '_' + sesion + '_' + categories[c][-1] +'.npy'
            im=np.load(impath)
            if im is None:
                pdb.set_trace()
            (h,w)=im.shape
            mask=im > -px
            
            #Files
            mom1[c]+=im[mask].sum()
            mom2[c]+=(im[mask]**2).sum()
            cont[c]+=mask.sum()
    mean=mom1/cont
    mom2=mom2/cont
    
    sigma=np.sqrt(mom2-mean**2)
    return categories,mean,sigma


if __name__ == "__main__":
    """ Arguments
        dir: directory where input dirs are saved
        input_dir: directory where data files are saved
        pixels: background pixels value
    """
    eConfig = {
        'dir':'RESNET',
        'input_dir':'ransac_TH_1.5_r_45',
        'pixels': '100'
        }
    
    # Arguments
    args = sys.argv[1::]
    for i in range(0,len(args),2):
        key = args[i]
        val = args[i+1]
        eConfig[key] = type(eConfig[key])(val)
    
    db_root = '../datasets'
    db_paths = ['../datasets/dataset_ssl', '../datasets/dataset_multicentrico/']
    out_db = '../datasets/dataset_global'

    if not os.path.exists(out_db):
        os.makedirs(out_db)
    
    pixels = int(eConfig['pixels'])
    #Indexing the regular dataset
    gannFile = out_db + '/patient_list.csv'
    global_dataset = None
    for i,db in enumerate(db_paths):
        annFile = db + '/patient_list.csv'
        dataset = pd.read_csv(annFile,dtype={'Sesion': str})
        if i==0:
            global_dataset=dataset
        else:
            global_dataset=pd.concat((global_dataset,dataset))
    global_dataset.to_csv(gannFile,index=False)
    
    #Normalization

    cats,mu,sigma = computeNorm(gannFile, db_root, px = pixels)
    normalization_data = {'categories': cats, 'mean': mu, 'std': sigma}
    print(normalization_data)
    data_path = os.path.join(out_db,'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    normalization_file = os.path.join(data_path,'normalization.npz')
    with open(normalization_file, 'wb') as f:
        pickle.dump(normalization_data, f)
  