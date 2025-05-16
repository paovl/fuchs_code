#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 19:27:41 2024

@authors: mariagonzalezgarcia, igonzalez
"""
import warnings
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
from dataAugmentation import Normalize, ToTensor, RandomRotation,RandomTranslation,centerMask,centerMinPAC, cropEye, reSize, PolarCoordinates, centerCrop, RandomJitter, Deactivate, CartesianCoordinates
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
import torch.nn as nn
import itertools

#train_model parameters are the network (model), the criterion (loss)
# the optimizer, a learning scheduler (una estrategia de lr strategy), and the training epochs
def train_model(model, image_datasets, mapList, dataloaders,criterion, optimizer, scheduler, device, results_path, normalizer, type, num_epochs=25,max_epochs_no_improvement=10,min_epochs=10, batchsize_train = None, batchsize_val = None, loss_type = 0):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0
    img1 = torch.zeros(batchsize_val,4, 141, 141)
    img2 = torch.zeros(batchsize_val, 4, 141, 141)
    best_loss = np.inf
    best_epoch = -1
    epochs_no_improvement = 0
    prev_lr=optimizer.param_groups[0]['lr']
    loss_values_train=[]
    loss_values_val =[]
    
    #Loop of epochs (each iteration involves train and val datasets)
    for epoch in range(num_epochs):
        valset = False
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        with open(text_file, "a") as file:
            file.write('\nEpoch {}/{}\n'.format(epoch, num_epochs - 1))
            file.write("-" * 10 + '\n')

        # Cada época tiene entrenamiento y validación
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set the model in training mode
                # batchsize = batchsize_train
            else:
                valset = True
                model.eval()   # Set the model in val mode (no grads)
                # batchsize = batchsize_val
              
            #Dataset size
            # pair_samples = len(image_datasets[phase])
            # numBatchs = pair_samples//batchsize
            # rest = pair_samples - (batchsize*numBatchs)
            
            # numSamples = 0 
            
            # if numBatchs>0:
            #     numSamples += math.comb(batchsize*2,2)*numBatchs
            
            # numSamples +=math.comb(rest*2,2)
                
            # Create variables to store outputs and labels
            outputs_m=[]
            labels_m=[]
            weights_loss_m = []
            weights_loss_batchnorm_m = []
            binary_weights_m = []
            sigmoid_weights_m = []
            running_loss = 0.0
            
            contSamples=0

            # Iterate (loop of batches)
            nbatch = 0
            for (img1, img2) in dataloaders[phase]:
                nbatch += 1
                # two batches, one has session 1 and the other session 2
                img_info = {'idx': img1['idx'],'patient':img1['patient'], 'eye': img1['eye'], 'session_1': img1['session'], 'session_2': img2['session']}

                # recuperar eyedataset idx
                idx_patient = img1['idx_patient']
                images_patient_idx = np.hstack((idx_patient, idx_patient))
                images_patient_idx = list(itertools.combinations(images_patient_idx, 2))

                # recuperar dataset idx
                img1_idx, img2_idx =  img1['idx'].cpu().numpy(), img2['idx'].cpu().numpy()
                images_idx = np.hstack((img1_idx, img2_idx))
                images_pairs_idx = list(itertools.combinations(images_idx, 2))

                # recuperar errores 
                error1_PAC, error2_PAC = img1['error_PAC'], img2['error_PAC']
                error1_BFS, error2_BFS = img1['error_BFS'], img2['error_BFS']

                # combinación de errores
                error1_PAC, error2_PAC = error1_PAC.cpu().numpy(), error2_PAC.cpu().numpy()
                error1_PAC, error2_PAC = np.nan_to_num(error1_PAC, 0),np.nan_to_num(error2_PAC, 0)
                error_PAC = np.hstack((error1_PAC, error2_PAC))
                error_PAC = list(itertools.combinations(error_PAC, 2))

                error_PAC_array = np.zeros((len(error_PAC), 2))
                for i in range(len(error_PAC)):
                    error_PAC_array[i, 0] = error_PAC[i][0]
                    error_PAC_array[i, 1] = error_PAC[i][1]

                error1_BFS, error2_BFS = error1_BFS.cpu().numpy(), error2_BFS.cpu().numpy()
                error1_BFS, error2_BFS = np.nan_to_num(error1_BFS, 0),np.nan_to_num(error2_BFS, 0)
                error_BFS = np.hstack((error1_BFS, error2_BFS))
                error_BFS= list(itertools.combinations(error_BFS, 2))

                error_BFS_array = np.zeros((len(error_BFS), 2))
                for i in range(len(error_BFS)):
                    error_BFS_array[i, 0] = error_BFS[i][0]
                    error_BFS_array[i, 1] = error_BFS[i][1]

                # recuperar imagenes
                bsize1, bsize2 = img1['bsize_x'], img2['bsize_x']
                img1, img2 = img1['img'].to(device), img2['img'].to(device)
                
                # copia de imágenes 
                img1_copy, img2_copy = img1.clone().detach(), img2.clone().detach()
                img1_copy_mean, img2_copy_mean = img1.clone().detach(), img2.clone().detach()
                images = torch.cat((img1, img2), dim=0)
                bsizes = torch.cat((bsize1, bsize2), dim=0)
    
                #Batch Size with new image pairs
                # batchSize = math.comb(images.shape[0],2)
                
                # Set grads to zero
                optimizer.zero_grad()

                # Forward
                # Register ops only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(images, bsizes) # Evaluate the model
                    output = F.normalize(output,dim=0)
                    
                    loss_contrastive, real_labels, pred_labels, weights_loss, weights_loss_batchnorm, binary_weights, sigmoid_weights, batchSize = criterion(output, error_PAC_array, loss_type = loss_type)
                    
                    # backward & parameters update only in train
                    if phase == 'train':
                        loss_contrastive.backward()
                        optimizer.step()
                
                # Accumulate the running loss
                running_loss += loss_contrastive.item() * batchSize
                
                # Store outputs and labels 
                outputs_m.extend(pred_labels.detach().cpu().numpy())

                labels_m.extend(real_labels.cpu().numpy())
                weights_loss_m.extend(weights_loss.cpu().numpy())
                weights_loss_batchnorm_m.extend(weights_loss_batchnorm.cpu().numpy())
                binary_weights_m.extend(binary_weights.cpu().numpy())
                sigmoid_weights_m.extend(sigmoid_weights.cpu().numpy())

                contSamples+=batchSize
                            
            #Accumulated loss by epoch
            epoch_loss = running_loss / contSamples
            
            if phase == 'train':
                loss_values_train.append(epoch_loss)
            else:
                loss_values_val.append(epoch_loss)
            
            outputs_m = np.array(outputs_m, dtype=float)
            labels_m = np.array(labels_m, dtype=int)
            weights_loss_norm_m = weights_loss_m / np.sum(weights_loss_m)
            weights_loss_m = np.array(weights_loss_m, dtype = float)
            weights_loss_norm_m = np.array(weights_loss_norm_m, dtype = float)
            weights_loss_batchnorm_m = np.array(weights_loss_batchnorm_m, dtype = float)
            binary_weights_m = np.array(binary_weights_m, dtype = float)
            sigmoid_weights_m = np.array(sigmoid_weights_m, dtype = float)
            
            #Compute the AUCs at the end of the epoch
            if loss_type == 0:
                auc = roc_auc_score(labels_m, outputs_m)
            elif loss_type == 1:
                auc = roc_auc_score(labels_m, outputs_m, sample_weight = weights_loss_norm_m)
            elif loss_type == 2:
                auc = roc_auc_score(labels_m, outputs_m, sample_weight = binary_weights_m)
            elif loss_type == 3:
                auc = roc_auc_score(labels_m, outputs_m, sample_weight = sigmoid_weights_m)
                
            #At the end of an epoch, update the lr scheduler    
            if phase == 'val':
            	
                scheduler.step(epoch_loss)
                if optimizer.param_groups[0]['lr']<prev_lr:
                    prev_lr=optimizer.param_groups[0]['lr']

            #And the Average AUC
            epoch_auc = auc

            print('{} Loss: {:.4f} AUC: {:.4f} lr: {}'.format(
                    phase, epoch_loss, auc, optimizer.param_groups[0]['lr']))
            
            with open(text_file, "a") as file:
                file.write('{} Loss: {:.4f} AUC: {:.4f}  lr: {} \n'.format(
                    phase, epoch_loss, auc,optimizer.param_groups[0]['lr']))

            # Deep copy of the best model
            if phase == 'val' and epoch>=min_epochs:
                if epoch_auc > best_auc and epoch_loss<best_loss: # or (epoch_auc==1.0 and epoch_loss<best_loss):
                    best_auc = epoch_auc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_no_improvement = 0
                    # torch.save(model.state_dict(), results_path + '/resnet18_self_supervised_fold.pth') # save the weights of the ResNet18
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

    with open(text_file, "a") as file:
        file.write('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        file.write('Best model in epoch {:02d} val AUC {:4f}\n'.format(best_epoch,best_auc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_auc


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # Config file
    eConfig = {
        'dir': 'try',
        'cfg_file':'conf.config1',
        'shift_da': -1,
        'angle_range': -1,
        'deactivate': -1,
        'th_weights': '0.0',
        'a_weights': '0.0',
        'seed':'0',
        'dataset_seed':'0',
        'type':'2',
        'delete_last_layer': '1',
        'fusion':'5',
        'max_radius': '7.0',
        'max_angle': '360.0',
        'loss_type':'0' # 0: original contrastive loss, 1: contrastive weighted loss, 2: contrastive binary weighted loss, 3: contrastive sigmoid weighted loss
        }

    eConfig['rings'] = [1.0,3.0,5.0]
    eConfig['arcs'] = [120.0, 240.0, 360.0]
    
    args = sys.argv[1::]
    i = 0
    while i < len(args) - 1:
        key = args[i]
        if key == 'rings' or key == 'arcs':
            eConfig[key] = []
            for j in range(i + 1, len(args)):
                val = args[j]
                if val.isdigit():
                    eConfig[key].append(float(val))
                else:
                    break
            i = j
        else:
            i = i + 1
            val = args[i]
            eConfig[key] = type(eConfig[key])(val)
            i = i + 1
          
    print('eConfig')
    print(eConfig)
    
    #Reading the config file
    cfg = importlib.import_module(eConfig['cfg_file'])

    eConfig['max_radius_px'] = cfg.imSize[0] / 2 # Max radius in px

    # Create results directory
    results_dir = 'results/' +  eConfig['dir']

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if len(cfg.mapList) == 14: 
        if not os.path.exists(results_dir + '/' + 'allMaps') :
            os.makedirs(results_dir + '/'+ 'allMaps')
        results_path = results_dir + '/' + 'allMaps/'
        weights_dir = results_dir + '/allMaps/weights'
    elif len(cfg.mapList) == 1:
        if not os.path.exists(results_dir + '/' + cfg.mapList[0]) :
            os.makedirs(results_dir + '/'+ cfg.mapList[0])
        results_path = results_dir + '/' + cfg.mapList[0] + '/'
        weights_dir = results_dir + '/' + cfg.mapList[0] + '/weights'
    else:
        if not os.path.exists(results_dir + '/' + '_'.join(cfg.mapList)) :
            os.makedirs(results_dir + '/'+ '_'.join(cfg.mapList))
        results_path = results_dir + '/' + '_'.join(cfg.mapList) + '/'
        weights_dir = results_dir + '/' + '_'.join(cfg.mapList) + '/weights'
    

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    text_file = results_path + '/'+ eConfig['dir'] + '.txt' 

    rseed = int(eConfig['seed'])
    if rseed>=0:
        cfg.random_seed = rseed
    
    dataset_seed = int(eConfig['dataset_seed'])
    if dataset_seed>=0:
        cfg.dataset_seed = dataset_seed
    
    rng = np.random.default_rng(seed=cfg.dataset_seed)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Reading conf from {}'.format(eConfig['cfg_file']))
    print(device)
    print('Random Seed: %d'%cfg.random_seed)

    with open(text_file, "w") as file:
        file.write('Reading conf from config --> {}'.format(eConfig['cfg_file']))
        file.write('Using {} device... \n' .format(device))
        file.write('Random seed = ' + eConfig['seed'])

    normalization_file = os.path.join(cfg.db_path,'dataset_global/data/normalization.npz')
    with open(normalization_file, 'rb') as f:
        normalization_data = pickle.load(f)
    
    #Filter using the map list
    idxCat = [normalization_data['categories'].index(cat) for cat in cfg.mapList]
    nMaps=len(cfg.mapList)
    
    if not hasattr(cfg, 'dropout'):
        cfg.dropout=0
    
    normalizer = Normalize(mean=normalization_data['mean'][idxCat],std=normalization_data['std'][idxCat])
    #First transforms
    if cfg.polar_coords:
        list_transforms=[reSize(cfg.imSize),
                         PolarCoordinates(),
                         ToTensor(),
                         normalizer
                         ]
    else:
        list_transforms=[reSize(cfg.imSize),
                         ToTensor(),
                         normalizer]
    
    if hasattr(cfg, 'cropEyeBorder'):
        cropEyeBorder=cfg.cropEyeBorder
    else:
        cropEyeBorder=-1
    
    if hasattr(cfg, 'centerMethod'):
        centerMethod=cfg.centerMethod
    else:
        centerMethod=0
        
    list_transforms=[cropEye(cfg.mapList,cropEyeBorder)] + list_transforms 
        
    if centerMethod == 1:
        list_transforms = [centerMask(cfg.mapList)] + list_transforms 
    elif centerMethod == 2:
        list_transforms = [centerMinPAC(cfg.mapList)] + list_transforms 

    list_transforms_basic = list_transforms
    #transform_chain_basic = transforms.Compose(list_transforms)
    
    #Now data augmentation
    if(float(eConfig["angle_range"]) != -1):
        angle_range = float(eConfig["angle_range"])
        list_transforms = [RandomRotation(angle_range, generator=rng)] + list_transforms
    else:
        if hasattr(cfg, 'angle_range_da'):
            list_transforms = [RandomRotation(cfg.angle_range_da, generator=rng)] + list_transforms 
    
    if(float(eConfig["shift_da"]) != -1):
        shift_da = float(eConfig["shift_da"])
        list_transforms = [RandomTranslation(shift_da, generator=rng)] + list_transforms 
    else:
        if hasattr(cfg, 'shift_da'):
            list_transforms = [RandomTranslation(cfg.shift_da, generator=rng)] + list_transforms 
    
    if not hasattr(cfg, 'jitter_brightness'):
        cfg.jitter_brightness = 0
    if not hasattr(cfg, 'jitter_contrast'):
        cfg.jitter_contrast = 0


    if cfg.jitter_brightness>0 or cfg.jitter_contrast>0:
        list_transforms = [RandomJitter(cfg.jitter_brightness, cfg.jitter_contrast, generator=rng)] + list_transforms 
    
    if(float(eConfig["deactivate"]) != -1):
        deactivate = int(eConfig["deactivate"])
        list_transforms = list_transforms + [Deactivate(deactivate)]
        list_transforms_basic = list_transforms_basic + [Deactivate(deactivate)]
    else:
        if hasattr(cfg, 'deactivate'):
            list_transforms = list_transforms + [Deactivate(cfg.deactivate)]
            list_transforms_basic = list_transforms_basic + [Deactivate(cfg.deactivate)]
    
    transform_chain = transforms.Compose(list_transforms)
    transform_chain_basic = transforms.Compose(list_transforms_basic)

    # Training iterations
    iterations = cfg.iterations

    total_scores=[]
    total_scores_norm=[]
    total_labels=[]
    total_ids = []
    AUCs = np.zeros((iterations,))

    # Dataset
    full_dataset = PairEyeDataset(cfg.csvFile, cfg.db_path, cfg.imageDir,cfg.dataDir, cfg.error_BFSDir, cfg.error_PACDir, transform=transform_chain,mapList=cfg.mapList,test=False,random_seed=int(eConfig['seed']),testProp=0)
    
    scores_matrix = np.full((len(full_dataset), iterations), np.nan)
    

    for iter in np.arange(iterations):

        random_seed = int(iter * 10) + int(eConfig['seed'])

        full_dataset = PairEyeDataset(cfg.csvFile, cfg.db_path, cfg.imageDir,cfg.dataDir, cfg.error_BFSDir, cfg.error_PACDir, transform=transform_chain,mapList=cfg.mapList,test=False,random_seed=random_seed,testProp=0)
        random_generator = torch.Generator().manual_seed(random_seed)
        dataset_sizes = len(full_dataset)*np.ones(cfg.numFolds) // float(cfg.numFolds)
        remSamples = len(full_dataset) - int(dataset_sizes.sum())
        for i in range(remSamples):
            dataset_sizes[i]+=1

        # Divide the dataset into folders
        fold_datasets = random_split(full_dataset,dataset_sizes.astype(int),generator=random_generator)

        print('\n\n' + '-'*5 + ' ITER %i ' %int(iter) + '-'*5 + '\n')
        with open(text_file, "a") as file:
            file.write('\n\n' + '-'*5 + ' ITER %i ' %int(iter) + '-'*5 + '\n\n')
        
        random.seed(random_seed)
        npr.seed(random_seed)

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if cfg.modelType!='Alexnet' and cfg.modelType!='VGG':
            torch.use_deterministic_algorithms(True,warn_only=True)

        # Create val and train dataset
        val_idx = 0
        val_dataset = copy.deepcopy(fold_datasets[val_idx])
        val_dataset.dataset.transform = transform_chain_basic
        train_idx = np.ones((cfg.numFolds,),dtype=int)
        train_idx[val_idx] = 0
        train_dataset = [fold_datasets[idx] for idx in np.nonzero(train_idx)[0]]
        train_dataset = ConcatDataset(train_dataset)

        # Specify training dataset, with a batch size of 8, shuffle the samples, and parallelize with 4 workers
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_bs,
                        shuffle=True, num_workers=cfg.train_num_workers, generator= random_generator)
        #Validation dataset => No shuffle
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.val_bs,
                        shuffle=False, num_workers=cfg.val_num_workers, generator= random_generator)

        if hasattr(cfg, 'modelType'):
            if cfg.modelType=='Resnet18':
                model_ft = Resnet18Model(nMaps,eConfig['rings'], eConfig['arcs'], float(eConfig['max_radius']), float(eConfig['max_angle']), int(eConfig['type']), int(eConfig['delete_last_layer']), int(eConfig['fusion']), pretrained=True, dropout=cfg.dropout)
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

        #The loss is Contrastive
        criterion = ContrastiveLoss(margin = 0.4, alpha=1.0, th_weights = float(eConfig['th_weights']), a_weights = float(eConfig['a_weights']))

        # We will use SGD with momentum as optimizer
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=cfg.lr, momentum=cfg.momentum,weight_decay=cfg.wd)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,factor=cfg.gamma,patience=cfg.step_size)

        image_datasets = {'train' : train_dataset, 'val': val_dataset}
        dataloaders = {'train' : train_dataloader, 'val': val_dataloader}
            
        model_ft, best_auc_model = train_model(model_ft, image_datasets, cfg.mapList, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                            device, results_path, normalizer, int(eConfig['type']), num_epochs=cfg.num_epochs, max_epochs_no_improvement=cfg.max_epochs_no_improvement,min_epochs=5, batchsize_train = cfg.train_bs, batchsize_val = cfg.val_bs, loss_type = int(eConfig['loss_type']))

        torch.save(model_ft.state_dict(), weights_dir + '/resnet18_self_supervised_iter' + str(iter) + '.pth')
    
        print("Best AUC model saved: ", best_auc_model)
        with open(text_file, "a") as file:
                file.write("Best AUC model saved" + str(best_auc_model))

    
