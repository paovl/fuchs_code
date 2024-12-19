#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:28:42 2024

@authors: mariagonzalezgarcia, igonzalez
"""

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
from sklearn.metrics import roc_auc_score, hamming_loss
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
from ContrastiveLoss import ContrastiveLoss

def train_model(model, image_datasets, dataloaders,criterion, optimizer, scheduler, device,num_epochs=25,max_epochs_no_improvement=10,min_epochs=10, batchsize_train = None, batchsize_val = None, best_auc_model = 0):
    
    since = time.time()
        
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0
    best_loss = np.inf
    best_epoch = -1
    epochs_no_improvement = 0
    prev_lr=optimizer.param_groups[0]['lr']
    
    #Loop of epochs (each iteration involves train and val datasets)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
        # every epoch has training and validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set the model in training mode
                batchsize = batchsize_train
            else:
                model.eval()   # Set the model in val mode (no grads)
                batchsize = batchsize_val

                        
            #Dataset size
            pair_samples = len(image_datasets[phase])
            numBatchs = pair_samples//batchsize
            rest = pair_samples - (batchsize*numBatchs)
            
            numSamples = 0 
            
            if numBatchs>0:
                numSamples += math.comb(batchsize*2,2)*numBatchs
            
            numSamples +=math.comb(rest*2,2)
                
            
            
            # Create variables to store outputs and labels
            outputs_m=np.zeros((numSamples,),dtype=float)
            labels_m=np.zeros((numSamples,),dtype=int)
            running_loss = 0.0
            
            contSamples=0

            # Iterate (loop of batches)
            for (img1, img2) in dataloaders[phase]:

                img1, img2= img1.to(device), img2.to(device)
                
                images = torch.cat((img1, img2), dim=0)
                
                #Batch Size with new image pairs
                batchSize = math.comb(images.shape[0],2)
                
                # Set grads to zero
                optimizer.zero_grad()

                # Forward
                # Register ops only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = F.normalize(model(images), dim=0)
                    
                    loss_contrastive, real_labels, pred_labels = criterion(output)
                    
                    # backward & parameters update only in train
                    if phase == 'train':
                        loss_contrastive.backward()
                        optimizer.step()
                
                # Accumulate the running loss
                running_loss += loss_contrastive.item() * batchSize
                
                
                # Store outputs and labels 
                outputs_m [contSamples:contSamples+batchSize]=pred_labels.detach().cpu().numpy()
                labels_m [contSamples:contSamples+batchSize]=real_labels.cpu().numpy()
                contSamples+=batchSize
                  
            #Accumulated loss by epoch
            epoch_loss = running_loss / numSamples
            
            #Compute the AUCs at the end of the epoch
            auc=roc_auc_score(labels_m, outputs_m)

            #At the end of an epoch, update the lr scheduler    
            if phase == 'val':
                # scheduler.step()
                scheduler.step(epoch_loss)
                if optimizer.param_groups[0]['lr']<prev_lr:
                    prev_lr=optimizer.param_groups[0]['lr']
                    

                
            #And the Average AUC
            epoch_auc = auc
            print('{} Loss: {:.4f} AUC: {:.4f} lr: {}'.format(
                phase, epoch_loss, auc,optimizer.param_groups[0]['lr']))
            
            
            # Deep copy of the best model
            if phase == 'val' and epoch>=min_epochs:
                if epoch_auc > best_auc and epoch_loss<best_loss: # or (epoch_auc==1.0 and epoch_loss<best_loss):
                    best_auc = epoch_auc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_no_improvement = 0
                    best_auc_model = best_auc
                elif epoch_loss<best_loss:
                    epochs_no_improvement=0
                    model.load_state_dict(best_model_wts)
                else:
                    epochs_no_improvement+=1
                    model.load_state_dict(best_model_wts)
                
                
        if epochs_no_improvement>=max_epochs_no_improvement:
            break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best model in epoch {:02d} val AUC {:4f}'.format(best_epoch,best_auc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_auc_model


if __name__ == '__main__':
    
    # TRAIN
    eConfig = {
        'cfg_file':'config',
        'seed':'-1'
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
    cfg=importlib.import_module(eConfig['cfg_file'])
    rseed=int(eConfig['seed'])
    if rseed>=0:
        cfg.random_seed=rseed
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    

    print('Random Seed: %d'%cfg.random_seed)
    normalization_file=os.path.join(cfg.dataDir,'normalization.npz')
    with open(normalization_file, 'rb') as f:
        normalization_data = pickle.load(f)
    
    #Filter using the map list
    idxCat = [normalization_data['categories'].index(cat) for cat in cfg.mapList]
    nMaps=len(cfg.mapList)
    
    
    if not hasattr(cfg, 'dropout'):
        cfg.dropout=0
    
    #First transforms
    if cfg.polar_coords:
        list_transforms=[reSize(cfg.imSize),
                         PolarCoordinates(),
                         ToTensor(),
                         Normalize(mean=normalization_data['mean'][idxCat],std=normalization_data['std'][idxCat])]
    else:
        list_transforms=[reSize(cfg.imSize),
                         ToTensor(),
                         Normalize(mean=normalization_data['mean'][idxCat],std=normalization_data['std'][idxCat])]
    
    if hasattr(cfg, 'cropEyeBorder'):
        cropEyeBorder=cfg.cropEyeBorder
    else:
        cropEyeBorder=-1
    
    if hasattr(cfg, 'centerMethod'):
        centerMethod=cfg.centerMethod
    else:
        centerMethod=0
        
    if cropEyeBorder>=0:
        list_transforms=[cropEye(cfg.mapList,cropEyeBorder)] + list_transforms 
        
    if centerMethod==1:
        list_transforms=[centerMask(cfg.mapList)] + list_transforms 
    elif centerMethod==2:
        list_transforms=[centerMinPAC(cfg.mapList)] + list_transforms 
    
    transform_chain_basic=transforms.Compose(list_transforms)
    
    #Now data augmentation
    if hasattr(cfg, 'angle_range_da'):
        list_transforms=[RandomRotation(cfg.angle_range_da)] + list_transforms 
    if hasattr(cfg, 'shift_da'):
        list_transforms=[RandomTranslation(cfg.shift_da)] + list_transforms 
    if not hasattr(cfg, 'jitter_brightness'):
        cfg.jitter_brightness=0
    if not hasattr(cfg, 'jitter_contrast'):
        cfg.jitter_contrast=0
    if cfg.jitter_brightness>0 or cfg.jitter_contrast>0:
        list_transforms=[RandomJitter(cfg.jitter_brightness, cfg.jitter_contrast)] + list_transforms 
    
    
    transform_chain=transforms.Compose(list_transforms)
    full_dataset=PairEyeDataset(cfg.csvFile, cfg.imageDir,cfg.dataDir, transform=transform_chain,mapList=cfg.mapList,test=False,random_seed=cfg.random_seed,testProp=0)
    generator1 = torch.Generator().manual_seed(cfg.random_seed)
    dataset_sizes=len(full_dataset)*np.ones(cfg.numFolds)//float(cfg.numFolds)
    remSamples=len(full_dataset)-int(dataset_sizes.sum())
    
    for i in range(remSamples):
       dataset_sizes[i]+=1
       
    #Divide the dataset into folders
    fold_datasets=random_split(full_dataset,dataset_sizes.astype(int),generator=generator1)
    
    total_scores=[]
    total_scores_norm=[]
    total_labels=[]
    AUCs=np.zeros((cfg.numFolds,)) 
    best_auc_model = 0
    best_margin = 0
    best_auc_overall = 0
    
    margins =[0.5, 1.5, 5, 10, 15, 20, 25] # if margin, change alpha for margin and alphas for margins
    alphas = np.arange(0.1,1.1,0.1)
    
    for alpha in alphas:
        
        print(f'Training with alpha={alpha}')
        total_auc = 0
    
        for fold in range(cfg.numFolds):
            print("Fold ",fold)
            
            # Set random seed for reproducibility
            random.seed(cfg.random_seed)
            npr.seed(cfg.random_seed)
            torch.manual_seed(cfg.random_seed)
            torch.backends.cudnn.benchmark = True
            if cfg.modelType!='Alexnet' and cfg.modelType!='VGG':
                torch.use_deterministic_algorithms(True,warn_only=True)
            
            
            val_idx=fold
            val_dataset=copy.deepcopy(fold_datasets[val_idx])
            val_dataset.dataset.transform=transform_chain_basic
            train_idx=np.ones((cfg.numFolds,),dtype=int)
            #train_idx[test_idx]=0
            train_idx[val_idx]=0
            train_dataset = [fold_datasets[idx] for idx in np.nonzero(train_idx)[0]]
            train_dataset = ConcatDataset(train_dataset)
            
            #Specify training dataset, with a batch size of 8, shuffle the samples, and parallelize with 4 workers
            train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_bs,
                            shuffle=False, num_workers=cfg.train_num_workers)
            #Validation dataset => No shuffle
            val_dataloader = DataLoader(val_dataset, batch_size=cfg.val_bs,
                            shuffle=False, num_workers=cfg.val_num_workers)
    
            if hasattr(cfg, 'modelType'):
                if cfg.modelType=='Resnet18':
                    model_ft = Resnet18Model(nMaps,pretrained=True,dropout=cfg.dropout)
                elif cfg.modelType=='Resnet34':
                    model_ft = Resnet34Model(nMaps,pretrained=True,dropout=cfg.dropout)
                elif cfg.modelType=='Resnet50':
                    model_ft = Resnet50Model(nMaps,pretrained=True,dropout=cfg.dropout)
                elif cfg.modelType=='Inception':
                    model_ft = InceptionModel(nMaps,pretrained=True,dropout=cfg.dropout)
                elif cfg.modelType=='RegnetY400m':
                    model_ft = RegnetY400mModel(nMaps,pretrained=True,dropout=cfg.dropout)
                elif cfg.modelType=='RegnetY800m':
                    model_ft = RegnetY800mModel(nMaps,pretrained=True,dropout=cfg.dropout)
                elif cfg.modelType=='RegnetY3_2g':
                    model_ft = RegnetY3_2gModel(nMaps,pretrained=True,dropout=cfg.dropout)
                elif cfg.modelType=='RegnetY32g':
                    model_ft = RegnetY32gModel(nMaps,pretrained=True,dropout=cfg.dropout)
                elif cfg.modelType=='Alexnet':
                    model_ft = AlexnetModel(nMaps,pretrained=True,dropout=cfg.dropout)
                elif cfg.modelType=='VGG':
                    model_ft = VGGModel(nMaps,pretrained=True,dropout=cfg.dropout)
                elif cfg.modelType=='Mobilev3large':
                    model_ft = Mobilev3large(nMaps,pretrained=True,dropout=cfg.dropout)
                    
            else:
                model_ft = Resnet18Model(nMaps,pretrained=True)
            
            if not hasattr(cfg, 'max_epochs_no_improvement'):
                cfg.max_epochs_no_improvement = 10
            
            
            model_ft = model_ft.to(device)
        
            #The loss is a cross-entropy loss
            criterion = ContrastiveLoss(margin = 0.5, alpha= alpha)
        
            # We will use SGD with momentum as optimizer
            optimizer_ft = optim.SGD(model_ft.parameters(), lr=cfg.lr, momentum=cfg.momentum,weight_decay=cfg.wd)
            # optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=cfg.lr, alpha=0.99, eps=1e-08, weight_decay=cfg.wd,momentum=cfg.momentum);#, momentum=0,
            # optimizer_ft = optim.Adam(model_ft.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.wd);#, momentum=0,
            # Our scheduler starts with an lr=1e-3 and decreases by a factor of 0.1 every 7 epochs.
            # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=cfg.step_size, gamma=cfg.gamma)
            exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,factor=cfg.gamma,patience=cfg.step_size)
            image_datasets = {'train' : train_dataset, 'val': val_dataset}
        
            dataloaders = {'train' : train_dataloader, 'val': val_dataloader}
                  
            model_ft, best_auc_model = train_model(model_ft, image_datasets, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                                   device, num_epochs=cfg.num_epochs, max_epochs_no_improvement=cfg.max_epochs_no_improvement,min_epochs=5, batchsize_train = cfg.train_bs, batchsize_val = cfg.val_bs)
            
            total_auc += best_auc_model
            
        avg_auc = total_auc / cfg.numFolds
        print(f"Alpha {alpha} average AUC: {avg_auc}")
        if avg_auc > best_auc_overall:
            best_auc_overall = avg_auc
            best_margin = alpha

        
    print("Best AUC model saved", best_auc_model)
    print(f'Best margin: {best_margin} with average AUC: {best_auc_overall}')
