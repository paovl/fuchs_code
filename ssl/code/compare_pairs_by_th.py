
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
import itertools

if __name__ == '__main__':
    
    eConfig = {
        'dir': 'deactivate70(full DB)',
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
        patients_val_batch = pickle.load(f)

    #patients_id_eyedataset = patients_val_batch['idx_patient'].cpu().numpy()
    #patients_id_eyedataset = np.hstack((patients_id_eyedataset, patients_id_eyedataset))
    #patients_id_eyedataset = list(itertools.combinations(patients_id_eyedataset, 2))
    patients_id_eyedataset = patients_val_batch['idx_patient']
    patients_id_eyedataset_array = np.zeros((len(patients_id_eyedataset), 2))
    for i in range(len(patients_id_eyedataset)):
        patients_id_eyedataset_array[i, 0] = patients_id_eyedataset[i][0]
        patients_id_eyedataset_array[i, 1] = patients_id_eyedataset[i][1]
    
    patients_id_dataset = patients_val_batch['idx']
    patients_id_dataset_array = np.zeros((len(patients_id_dataset), 2))
    for i in range(len(patients_id_dataset)):
        patients_id_dataset_array[i, 0] = patients_id_dataset[i][0]
        patients_id_dataset_array[i, 1] = patients_id_dataset[i][1]
    
    patients_labels = patients_val_batch['label']
    patients_outputs = patients_val_batch['output']

    idxCat = [normalization_data['categories'].index(cat) for cat in cfg.mapList]

    list_transforms=[cropEye(cfg.mapList,1),
                        reSize(cfg.imSize),
                         ToTensor()]
    
    transform_chain = transforms.Compose(list_transforms)
    full_dataset = PairEyeDataset(cfg.csvFile, cfg.db_path, cfg.imageDir,cfg.dataDir, cfg.error_BFSDir, cfg.error_PACDir, transform = transform_chain, mapList=cfg.mapList,test=False,random_seed=cfg.random_seed,testProp=0)

    patients_eye1 = full_dataset.dataset.iloc[patients_id_dataset_array[:,0].astype(int)]['Ojo'].values
    patients_eye2 = full_dataset.dataset.iloc[patients_id_dataset_array[:,1].astype(int)]['Ojo'].values

    patients_session1 = full_dataset.dataset.iloc[patients_id_dataset_array[:,0].astype(int)]['Sesion'].values
    patients_session2 = full_dataset.dataset.iloc[patients_id_dataset_array[:,1].astype(int)]['Sesion'].values

    patients_id1 = full_dataset.dataset.iloc[patients_id_dataset_array[:,0].astype(int)]['Patient'].values
    patients_id2 = full_dataset.dataset.iloc[patients_id_dataset_array[:,1].astype(int)]['Patient'].values
    
    image_path_images = dir_path +'/comparison_images'
    
    if not os.path.exists(image_path_images):
            os.makedirs(image_path_images)

    image_path_FP = image_path_images +'/false_positives'
    image_path_TP = image_path_images +'/true_positives'
    image_path_FN = image_path_images +'/false_negatives'
    image_path_TN = image_path_images +'/true_negatives'

    if not os.path.exists(image_path_FP):
            os.makedirs(image_path_FP)
        
    if not os.path.exists(image_path_TP):
            os.makedirs(image_path_TP)
    
    if not os.path.exists(image_path_FN):
            os.makedirs(image_path_FN)
    
    if not os.path.exists(image_path_TN):
            os.makedirs(image_path_TN)

    len_dataset = len(full_dataset)

    for idx in np.arange(0, patients_id_eyedataset_array.shape[0]):
        idx_eyedataset_img1 = int(patients_id_eyedataset_array[idx, 0])
        idx_eyedataset_img2 = int(patients_id_eyedataset_array[idx, 1])

        idx_dataset_img1 = int(patients_id_dataset_array[idx, 0])
        idx_dataset_img2 = int(patients_id_dataset_array[idx, 1])

        img1_session = patients_session1[idx]
        img2_session = patients_session2[idx]

        img1_eye = patients_eye1[idx]
        img2_eye = patients_eye2[idx]

        img1_patient = patients_id1[idx]
        img2_patient = patients_id2[idx]

        output = patients_outputs[idx]
        label = patients_labels[idx]

        # Extraemos primera imagen
        img1_1, img1_2 = full_dataset[idx_eyedataset_img1]

        if (img1_1['session'] == img1_session):
            img1 = img1_1['img']
        else:
            img1 = img1_2['img']
        
        # Extraemos segunda imagen
        img2_1, img2_2 = full_dataset[idx_eyedataset_img2]

        if (img2_1['session'] == img2_session):
            img2 = img2_1['img']
        else:
            img2 = img2_2['img']
        
        img1_mask = np.where(img1<0)
        img1[img1_mask] = 0
        
        img1 = img1.cpu().numpy()  

        img2_mask = np.where(img2<0)
        img2[img2_mask] = 0

        img2 = img2.cpu().numpy()

        fig, axs = plt.subplots(img1.shape[0], 3, figsize=(16, 8)) 
  
        for i in range(img1.shape[0]):

            img1_channel = img1[i, :, :]
            img2_channel = img2[i, :, :]

            axs[i, 0].imshow(img1_channel)
            axs[i, 0].axis('off')
            axs[i, 0].set_title(str(img1_patient)+ '_' + img1_session + ' - ' +cfg.mapList[i])

            axs[i, 1].imshow(img2_channel)
            axs[i, 1].axis('off')
            axs[i, 1].set_title(str(img2_patient)+ '_' + img2_session + ' - ' +cfg.mapList[i])

            img_diff_channel = cv2.subtract(img1_channel, img2_channel)

            axs[i, 2].imshow(img_diff_channel)
            axs[i, 2].axis('off')
            axs[i, 2].set_title('Img diff - ' +cfg.mapList[i])

        plt.tight_layout()

        # graphic to numpy array 
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) 

        plt.close(fig)
        plt.imshow(data)
        plt.axis('off')
        plt.title(str(img1_patient) + '_' + img1_eye + '_' + img1_session + ' - '+ str(img2_patient) + '_' + img2_eye + '_' + img2_session + "\nScore = " + str(round(output, 4)) + ", Label = " + str(label))
        if (int(label) == 0):
            if (output < 0.5):
                image_path = image_path_TN
            else:
                image_path = image_path_FP
        else:
            if (output >= 0.5):
                image_path = image_path_TP
            else:
                image_path = image_path_FN

        plt.savefig(image_path + '/'+ str(img1_patient) + '_' + img1_eye + '_' + img1_session + ' - '+ str(img2_patient) + '_' + img2_eye + '_' + img2_session + '.png', bbox_inches='tight', pad_inches=0.1)
        plt.clf()
 