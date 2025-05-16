
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

if __name__ == '__main__':
    
    eConfig = {
        'dir': 'default',
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
    normalization_file = os.path.join(cfg.dataDir,'normalization.npz')
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
    full_dataset = PairEyeDataset(cfg.csvFile, cfg.imageDir,cfg.dataDir, cfg.error_BFSDir, cfg.error_PACDir, transform = transform_chain, mapList=cfg.mapList,test=False,random_seed=cfg.random_seed,testProp=0)

    image_path_diff = dir_path +'/comparison_diff'
    image_path_pairs = dir_path +'/comparison_pairs'
    
    if not os.path.exists(image_path_diff):
            os.makedirs(image_path_diff)

    if not os.path.exists(image_path_pairs):
            os.makedirs(image_path_pairs)

    len_dataset = len(full_dataset)

    for idx in patients_batch:
        im1, im2 = full_dataset[idx]
        
        img1 = im1['img']
        img1_mask = np.where(img1<0)
        img1[img1_mask] = 0
        
        img1 = img1.cpu().numpy()  

        img2 = im2['img']
        img2_mask = np.where(img2<0)
        img2[img2_mask] = 0

        img2 = img2.cpu().numpy()

        fig, axs = plt.subplots(img1.shape[0], 3, figsize=(16, 8)) 
  
        for i in range(img1.shape[0]):

            img1_channel = img1[i, :, :]
            img2_channel = img2[i, :, :]

            axs[i, 0].imshow(img1_channel)
            axs[i, 0].axis('off')
            axs[i, 0].set_title('Img1 - ' +cfg.mapList[i])

            axs[i, 1].imshow(img2_channel)
            axs[i, 1].axis('off')
            axs[i, 1].set_title('Img2 - ' + cfg.mapList[i])

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
        plt.title(str(im2['patient']) + '_' + im2['eye'] + '_' + im1['session'] + '_'+ im2['session'])
        plt.savefig(image_path_pairs + '/'+ str(im2['patient']) + '_' + im2['eye'] + '_' + im1['session'] + '_'+ im2['session']+ '.png', bbox_inches='tight', pad_inches=0.1)
        plt.clf()
    
    for idx in patients_batch:
        im1, im2 = full_dataset[idx]
        
        img1 = im1['img']
        img1_mask = np.where(img1<0)
        img1[img1_mask] = 0
        
        img1 = img1.cpu().numpy()  

        for idx2 in patients_batch:
            im_not, im1_2 = full_dataset[idx2]

            if(idx2 != idx):
                fig, axs = plt.subplots(img1.shape[0], 3, figsize=(16, 8)) 

                img1_2 = im1_2['img']
                img1_2_mask = np.where(img1_2<0)
                img1_2[img1_2_mask] = 0

                img1_2 = img1_2.cpu().numpy()

                for i in range(img1.shape[0]):

                    img1_channel = img1[i, :, :]
                    img1_2_channel = img1_2[i, :, :]

                    axs[i, 0].imshow(img1_channel)
                    axs[i, 0].axis('off')
                    axs[i, 0].set_title(str(im1['patient']) + '_' + im1['session']+ ' - ' +cfg.mapList[i])

                    axs[i, 1].imshow(img1_2_channel)
                    axs[i, 1].axis('off')
                    axs[i, 1].set_title(str(im1_2['patient']) + '_' + im1_2['session']+ ' - ' +cfg.mapList[i])
        
                    img_diff_channel = cv2.subtract(img1_channel, img1_2_channel)

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
                plt.title(str(im1['patient']) + '_' + im1['eye'] + '_' + im1['session'] + ' - '+ str(im1_2['patient']) + '_' + im1_2['eye'] + '_' + im1_2['session'])
                plt.savefig(image_path_diff + '/'+ str(im1['patient']) + '_' + im1['eye'] + '_' + im1['session'] + ' - '+ str(im1_2['patient']) + '_' + im1_2['eye'] + '_' + im1_2['session'] + '.png', bbox_inches='tight', pad_inches=0.1)
                plt.clf()
        
        img2 = im2['img']
        img2_mask = np.where(img2<0)
        img2[img2_mask] = 0

        img2 = img2.cpu().numpy()
        
        for idx2 in patients_batch:
            im2_1, im_not= full_dataset[idx2]

            if(idx2 != idx):
                fig, axs = plt.subplots(img1.shape[0], 3, figsize=(16, 8))

                img2_1 = im2_1['img']
                img2_1_mask = np.where(img2_1<0)
                img2_1[img2_1_mask] = 0

                img2_1 = img2_1.cpu().numpy()

                for i in range(img2.shape[0]):

                    img2_channel = img2[i, :, :]
                    img2_1_channel = img2_1[i, :, :]

                    axs[i, 0].imshow(img2_channel)
                    axs[i, 0].axis('off')
                    axs[i, 0].set_title(str(im2['patient']) + '_' + im2['session']+ ' - ' +cfg.mapList[i])

                    axs[i, 1].imshow(img2_1_channel)
                    axs[i, 1].axis('off')
                    axs[i, 1].set_title(str(im2_1['patient']) + '_' + im2_1['session']+ ' - ' +cfg.mapList[i])
        
                    img_diff_channel = cv2.subtract(img2_channel, img2_1_channel)

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
                plt.title(str(im2['patient']) + '_' + im2['eye'] + '_' + im2['session'] + ' - '+ str(im2_1['patient']) + '_' + im2_1['eye'] + '_' + im2_1['session'])
                plt.savefig(image_path_diff + '/'+ str(im2['patient'])+ '_' + im2['eye'] + '_' + im2['session'] + ' - '+ str(im2_1['patient']) + '_' + im2_1['eye'] + '_' + im2_1['session'] + '.png', bbox_inches='tight', pad_inches=0.1)
                plt.clf()
    # repeat
    
    for idx in patients_batch:
        im1, im2 = full_dataset[idx]
        
        img1 = im1['img']
        img1_mask = np.where(img1<0)
        img1[img1_mask] = 0
        
        img1 = img1.cpu().numpy()  

        for idx2 in patients_batch:
            im1_2, im_not = full_dataset[idx2]

            if(idx2 != idx):
                fig, axs = plt.subplots(img1.shape[0], 3, figsize=(16, 8)) 

                img1_2 = im1_2['img']
                img1_2_mask = np.where(img1_2<0)
                img1_2[img1_2_mask] = 0

                img1_2 = img1_2.cpu().numpy()

                for i in range(img1.shape[0]):

                    img1_channel = img1[i, :, :]
                    img1_2_channel = img1_2[i, :, :]

                    axs[i, 0].imshow(img1_channel)
                    axs[i, 0].axis('off')
                    axs[i, 0].set_title(str(im1['patient']) + '_' + im1['session']+ ' - ' +cfg.mapList[i])

                    axs[i, 1].imshow(img1_2_channel)
                    axs[i, 1].axis('off')
                    axs[i, 1].set_title(str(im1_2['patient']) + '_' + im1_2['session']+ ' - ' +cfg.mapList[i])
        
                    img_diff_channel = cv2.subtract(img1_channel, img1_2_channel)

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
                plt.title(str(im1['patient']) + '_' + im1['eye'] + '_' + im1['session'] + ' - '+ str(im1_2['patient']) + '_' + im1_2['eye'] + '_' + im1_2['session'])
                plt.savefig(image_path_diff + '/'+ str(im1['patient']) + '_' + im1['eye'] + '_' + im1['session'] + ' - '+ str(im1_2['patient']) + '_' + im1_2['eye'] + '_' + im1_2['session'] + '.png', bbox_inches='tight', pad_inches=0.1)
                plt.clf()
        
        img2 = im2['img']
        img2_mask = np.where(img2<0)
        img2[img2_mask] = 0

        img2 = img2.cpu().numpy()
        
        for idx2 in patients_batch:
            im_not, im2_1= full_dataset[idx2]

            if(idx2 != idx):
                fig, axs = plt.subplots(img1.shape[0], 3, figsize=(16, 8))

                img2_1 = im2_1['img']
                img2_1_mask = np.where(img2_1<0)
                img2_1[img2_1_mask] = 0

                img2_1 = img2_1.cpu().numpy()

                for i in range(img2.shape[0]):

                    img2_channel = img2[i, :, :]
                    img2_1_channel = img2_1[i, :, :]

                    axs[i, 0].imshow(img2_channel)
                    axs[i, 0].axis('off')
                    axs[i, 0].set_title(str(im2['patient']) + '_' + im2['session']+ ' - ' +cfg.mapList[i])

                    axs[i, 1].imshow(img2_1_channel)
                    axs[i, 1].axis('off')
                    axs[i, 1].set_title(str(im2_1['patient']) + '_' + im2_1['session']+ ' - ' +cfg.mapList[i])
        
                    img_diff_channel = cv2.subtract(img2_channel, img2_1_channel)

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
                plt.title(str(im2['patient']) + '_' + im2['eye'] + '_' + im2['session'] + ' - '+ str(im2_1['patient']) + '_' + im2_1['eye'] + '_' + im2_1['session'])
                plt.savefig(image_path_diff + '/'+ str(im2['patient'])+ '_' + im2['eye'] + '_' + im2['session'] + ' - '+ str(im2_1['patient']) + '_' + im2_1['eye'] + '_' + im2_1['session'] + '.png', bbox_inches='tight', pad_inches=0.1)
                plt.clf()
    





