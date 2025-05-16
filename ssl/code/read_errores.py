
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
from PIL import Image
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader,random_split,ConcatDataset
from torchvision import transforms, utils, models
from skimage import io, transform, color, morphology, util
from PairEyeDataset import PairEyeDataset
from sklearn.metrics import roc_auc_score, hamming_loss, confusion_matrix
import numpy as np
import cv2
import pdb
import pickle
from dataAugmentation import Normalize, ToTensor, RandomRotation,RandomTranslation,centerMask,centerMinPAC, cropEye, reSize, PolarCoordinates, centerCrop, RandomJitter
import time
import copy
from models import Resnet18Model,Resnet34Model,Resnet50Model,InceptionModel,AlexnetModel,VGGModel,Mobilev3large
from models import RegnetY400mModel,RegnetY800mModel,RegnetY3_2gModel,RegnetY32gModel
import random
import numpy.random as npr
import sys
import importlib
import matplotlib.pyplot as plt
from models import Resnet18Model
from code.ContrastiveLoss_here import ContrastiveLoss
import torch.nn as nn
import pandas as pd

def print_data(data):
    # Title
    print('{:<20}'.format('patient'), end='')
    for key in data.keys():
        if key != 'patient':
            print('{:<20}'.format(key), end='')
    print()
    
    # Data
    for i in range(len(data['patient'])):
        print('{:<20}'.format(data['patient'][i]), end='')
        for key in data.keys():
            if key != 'patient':
                print('{:<20.8f}'.format(data[key][i]), end='')

def write_data(file_path, data):
     with open(file_path, "w") as file:
        # Title
        file.write('{:<20}'.format('patient'))
        for key in data.keys():
            if key != 'patient':
                file.write('{:<20}'.format(key))
        file.write('\n')
    
        # Data
        for i in range(len(data['patient'])):
            file.write('{:<20}'.format(data['patient'][i]))
            for key in data.keys():
                if key != 'patient':
                    file.write('{:<20.8f}'.format(data[key][i]))
            file.write('\n')

if __name__ == '__main__':
    
    eConfig = {
        'dir': 'gray results/deactivate70',
        'cfg_file':'conf.config1',
        }
  
    args = sys.argv[1::]
    for i in range(0,len(args),2):
        key = args[i]
        val = args[i+1]
        eConfig[key] = type(eConfig[key])(val)
        print (str(eConfig[key]))
          
    print('args')
    print(sys.argv)
    print('eConfig')
    print(eConfig)

    print('Reading conf from {}'.format(eConfig['cfg_file']))
    
    text_file_PAC = './errors_by_PAC.txt'
    text_file_BFS = './errors_by_BFS.txt'
    #Reading the config file
    cfg = importlib.import_module(eConfig['cfg_file'])
    normalization_file = os.path.join(cfg.db_path,'dataset_global/data/normalization.npz')
    patients_val_batch = 'val_batch_patients.pkl'

    if len(cfg.mapList) == 14: 
        dir_path = 'results/' + eConfig['dir'] + '/allMaps'
    elif len(cfg.mapList) == 1:
        dir_path = 'results/' + eConfig['dir'] + '/' + cfg.mapList[0]
    else:
        dir_path = 'results/' + eConfig['dir'] + '/' + '_'.join(cfg.mapList)

    with open(normalization_file, 'rb') as f:
        normalization_data = pickle.load(f)

    with open(dir_path + '/' + patients_val_batch, 'rb') as f:
        patients_batch = pickle.load(f)

    patients_batch = patients_batch['idx_patient'].cpu().numpy()
    idxCat = [normalization_data['categories'].index(cat) for cat in cfg.mapList]

    list_transforms=[reSize(cfg.imSize),
                         ToTensor()]
    
    transform_chain = transforms.Compose(list_transforms)
    full_dataset = PairEyeDataset(cfg.csvFile, cfg.db_path, cfg.imageDir,cfg.dataDir, cfg.error_BFSDir, cfg.error_PACDir, transform = transform_chain, mapList=cfg.mapList,test=False,random_seed=cfg.random_seed,testProp=0)

    patients = []
    eyes = []
    sessions = []
    errors_PAC = []
    errors_BFS = []
    for i in range(0, len(full_dataset)):
        muestra = full_dataset[i]

        muestra_1 = muestra [0]
        patients.append(str(muestra_1['patient'])+ '_'+ muestra_1['eye'] + '_'+ muestra_1['session'])
        eyes.append(muestra_1['eye'])
        sessions.append(muestra_1['session'])
        errors_PAC.append(muestra_1['error_PAC'])
        errors_BFS.append(muestra_1['error_BFS'])

        muestra_2 = muestra [1]
        patients.append(str(muestra_2['patient']) + '_'+ muestra_2['eye'] + '_'+ muestra_2['session'])
        eyes.append(muestra_2['eye'])
        sessions.append(muestra_2['session'])
        errors_PAC.append(muestra_2['error_PAC'])
        errors_BFS.append(muestra_2['error_BFS'])

    # plot histogram
    plt.hist(errors_PAC, bins=80, edgecolor='black')
    plt.title("Histograma (error PAC)")
    plt.xlabel("Valores")
    plt.ylabel("Frecuencia")
    plt.savefig('histogram_PAC.png', bbox_inches='tight', pad_inches=0.1)
    plt.clf()

    # plot histogram
    plt.hist(errors_BFS, bins=80, edgecolor='black')
    plt.title("Histograma (error BFS)")
    plt.xlabel("Valores")
    plt.ylabel("Frecuencia")
    plt.savefig('histogram_BFS.png', bbox_inches='tight', pad_inches=0.1)
    plt.clf()

    df = pd.DataFrame({ 'patient': patients, 'error_PAC': errors_PAC, 'error_BFS': errors_BFS})
    df_sorted_by_PAC = df.sort_values(by='error_PAC', ascending=False).reset_index(drop=True)
    print_data(df_sorted_by_PAC)
    write_data(text_file_PAC, df_sorted_by_PAC)

    df_sorted_by_BFS = df.sort_values(by='error_BFS', ascending=False).reset_index(drop=True)
    print_data(df_sorted_by_BFS)
    write_data(text_file_BFS, df_sorted_by_BFS)