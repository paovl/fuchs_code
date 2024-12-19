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
from dataAugmentation import Normalize, ToTensor, RandomRotation,RandomTranslation,centerMask,centerMinPAC, cropEye, reSize, PolarCoordinates, centerCrop, RandomJitter, Deactivate, cropEyePolar
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

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for saving activations and gradients
        self.target_layer.register_forward_hook(self.save_activation) # capture activations in forward pass
        self.target_layer.register_backward_hook(self.save_gradient) # capture layer gradients in backward pass

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach() # grad from this

    def __call__(self, input_tensor1, input_tensor2):

        embedding1 = self.model(input_tensor1) # forward
        embedding2 = self.model(input_tensor2) # forward
        
        distance = torch.norm(embedding1 - embedding2, p=2)
        # distances = torch.norm(embedding1 - embedding2, p=2, dim=1)
        # distance = distances.mean()
        
        # Backward to compute gradients
        self.model.zero_grad() # reset gradients
        distance.backward(retain_graph=True) # compute gradients with respect to all parameters

        # Gradients' average
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3]) # Mean of last layer gradients (after batch norm) = Result one gradient for each channel

        # Weights
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # Heatmap
        heatmap = torch.mean(abs(self.activations), dim=1).squeeze()
        # heatmap = F.relu(heatmap)
        # heatmap = abs(heatmap)
        heatmap /= torch.max(heatmap)

        return heatmap.cpu().numpy()

#train_model parameters are the network (model), the criterion (loss)
# the optimizer, a learning scheduler (una estrategia de lr strategy), and the training epochs
def train_model(model, image_datasets, mapList, dataloaders,criterion, optimizer, scheduler, device, results_path, normalizer, num_epochs=25,max_epochs_no_improvement=10,min_epochs=10, batchsize_train = None, batchsize_val = None, best_auc_model = 0, grad_cam_status = False, plot_image_best = [], save_plot_best = [], plot_batch_image_best = [], save_plot_batch_best = [], patients_val_batch = []):
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
    
    target_layer = next(layer for layer in reversed(list(model.modules())) if isinstance(layer, nn.BatchNorm2d)) #last layer of the model
    grad_cam = GradCAM(model, target_layer)
    
    #Loop of epochs (each iteration involves train and val datasets)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        with open(text_file, "a") as file:
            file.write('\nEpoch {}/{}\n'.format(epoch, num_epochs - 1))
            file.write("-" * 10 + '\n')

        # Cada época tiene entrenamiento y validación
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

                # two batches, one has session 1 and the other session 2
                img_info = {'idx': img1['idx'],'patient':img1['patient'], 'eye': img1['eye'], 'session_1': img1['session'], 'session_2': img2['session']}

                idx_patient = img1['idx_patient']

                # recuperar errores 
                error1_PAC, error2_PAC = img1['error_PAC'], img2['error_PAC']
                error1_BFS, error2_BFS = img1['error_BFS'], img2['error_BFS']

                # recuperar idx
                img1_idx, img2_idx =  img1['idx'].cpu().numpy(), img2['idx'].cpu().numpy()

                # recuperar imagenes
                img1, img2 = img1['img'].to(device), img2['img'].to(device)

                # copia de imágenes 
                img1_copy, img2_copy = img1.clone().detach(), img2.clone().detach()
                img1_copy_mean, img2_copy_mean = img1.clone().detach(), img2.clone().detach()

                # combinacion de imagenes 
                images = torch.cat((img1, img2), dim=0)

                # combinacion de indices
                images_idx = np.hstack((img1_idx, img2_idx))
                images_pairs_idx = list(itertools.combinations(images_idx, 2))

                # combinación de errores
                error1_PAC, error2_PAC = error1_PAC.cpu().numpy(), error2_PAC.cpu().numpy()
                error1_PAC, error2_PAC = np.nan_to_num(error1_PAC, 0),np.nan_to_num(error2_PAC, 0)
                error_PAC = np.hstack((error1_PAC, error2_PAC)) * 100
                error_PAC = list(itertools.combinations(error_PAC, 2))

                error_PAC_array = np.zeros((len(error_PAC), 2))
                for i in range(len(error_PAC)):
                    error_PAC_array[i, 0] = error_PAC[i][0]
                    error_PAC_array[i, 1] = error_PAC[i][1]

                error1_BFS, error2_BFS = error1_BFS.cpu().numpy(), error2_BFS.cpu().numpy()
                error1_BFS, error2_BFS = np.nan_to_num(error1_BFS, 0),np.nan_to_num(error2_BFS, 0)
                error_BFS = np.hstack((error1_BFS, error2_BFS)) * 100
                error_BFS= list(itertools.combinations(error_BFS, 2))

                error_BFS_array = np.zeros((len(error_BFS), 2))
                for i in range(len(error_BFS)):
                    error_BFS_array[i, 0] = error_BFS[i][0]
                    error_BFS_array[i, 1] = error_BFS[i][1]
    
                #Batch Size with new image pairs
                batchSize = math.comb(images.shape[0],2)
                
                # Set grads to zero
                optimizer.zero_grad()

                # Forward
                # Register ops only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = F.normalize(model(images),dim=0)
                    
                    loss_contrastive, real_labels, pred_labels = criterion(output, error_PAC_array, error_BFS_array)
                    
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
            
            if phase == 'train':
                loss_values_train.append(epoch_loss)
            else:
                loss_values_val.append(epoch_loss)

            #Compute the AUCs at the end of the epoch
            auc = roc_auc_score(labels_m, outputs_m)
            
            #At the end of an epoch, update the lr scheduler    
            if phase == 'val':

                #predicted_labels = (pred_labels >= 0.7).long()
                #conf_matrix = confusion_matrix(real_labels.cpu().numpy(),predicted_labels.cpu().numpy())
                #print("Confusion matrix")
                #print(conf_matrix)
            	
                scheduler.step(epoch_loss)
                if optimizer.param_groups[0]['lr']<prev_lr:
                    prev_lr=optimizer.param_groups[0]['lr']
                    

            #And the Average AUC
            epoch_auc = auc
            print('{} Loss: {:.4f} AUC: {:.4f} lr: {}'.format(
                phase, epoch_loss, auc,optimizer.param_groups[0]['lr']))
            
            with open(text_file, "a") as file:
                file.write('{} Loss: {:.4f} AUC: {:.4f} lr: {} \n'.format(
                    phase, epoch_loss, auc,optimizer.param_groups[0]['lr']))

            
            # Deep copy of the best model
            if phase == 'val' and epoch>=min_epochs:
                if epoch_auc > best_auc and epoch_loss<best_loss: # or (epoch_auc==1.0 and epoch_loss<best_loss):
                    best_auc = epoch_auc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_no_improvement = 0
                    if best_auc > best_auc_model:
                        torch.save(model.state_dict(), results_path + '/resnet18_self_supervised_fold.pth') # save the weights of the ResNet18
                        best_auc_model = best_auc

                        # Grad-CAM in validation
                        if grad_cam_status == True and phase == 'val':

                            plot_array = []
                            plot_batch_array = []
                            save_plot_array = []
                            save_plot_batch_array = []
                            patients_val_batch = []

                            heatmaps = grad_cam(img1, img2)
                            
                            # Mean heatmap for batch
                            heatmap_mean = np.mean(heatmaps[:,:,:], axis=0)

                            # for each image
                            for i in np.arange(img1.shape[0]):
                                plot_image = save_gradcam_images(img1_copy[i,:,:,:], img2_copy[i,:,:,:], heatmaps[i,:,:], mapList, {key: value[i] for key, value in img_info.items()}, normalizer)
                                plot_array.append(plot_image)
                                save_plot_array.append(str(img_info['patient'][i].cpu().numpy())+ '_' + str(img_info['eye'][i]) + '.png')

                            for i in np.arange(img1.shape[0]):
                                plot_batch_image = save_gradcam_mean_images(img1_copy_mean[i,:,:,:], img2_copy_mean[i,:,:,:], heatmap_mean, mapList, {key: value[i] for key, value in img_info.items()}, normalizer)
                                plot_batch_array.append(plot_batch_image)
                                patient = str(img_info['patient'][i].cpu().numpy())
                                save_plot_batch_array.append(str(img_info['patient'][i].cpu().numpy())+ '_' + str(img_info['eye'][i]) + '.png')

                        patients_val_batch = {'idx_patient':idx_patient, 'idx':images_pairs_idx, 'label':labels_m, 'output': outputs_m}
                        plot_image_best = plot_array
                        save_plot_best = save_plot_array
                        plot_batch_image_best = plot_batch_array
                        save_plot_batch_best = save_plot_batch_array

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
    return model, best_auc_model, plot_image_best, save_plot_best, plot_batch_image_best, save_plot_batch_best, patients_val_batch

def save_gradcam_images(img1, img2, heatmap, mapList, img_info, normalizer):
    img1, img2= normalizer.denormalize((img1, img2))
    alpha = 0.6
    colormap = cv2.COLORMAP_JET
    fig, axs = plt.subplots(2, img1.shape[0] *2, figsize=(16, 8))  # 2 filas (una por imagen), 4 columnas (una por canal)

    # Resize to have the same size as the image
    heatmap_resize = cv2.resize(heatmap, (img1.shape[2], img1.shape[1]))

    # Heatmap normalized between 0 and 255
    heatmap_resize = np.uint8(255 * heatmap_resize)

    heatmap_resize = cv2.applyColorMap(heatmap_resize, colormap)

    img1 = img1.permute(1, 2, 0).cpu().numpy()  # Change [C, H, W] to [H, W, C]
    img2 = img2.permute(1, 2, 0).cpu().numpy()  

    # plot original
    idx = 0

    for i in range(img1.shape[-1]):

        img1_channel = (img1[ :, :, i] * 255).astype(np.uint8)
        img2_channel = (img2[ :, :, i] * 255).astype(np.uint8)

        axs[0, idx].imshow(img1_channel, cmap='gray')
        axs[0, idx].axis('off')
        axs[0, idx].set_title(mapList[i])

        axs[1, idx].imshow(img2_channel, cmap='gray')
        #axs[1, i].imshow(heatmap)
        axs[1, idx].axis('off')
        axs[1, idx].set_title(mapList[i])

        idx = idx + 1

        img1_channel_rgb= cv2.cvtColor(img1_channel, cv2.COLOR_GRAY2BGR)
        img2_channel_rgb= cv2.cvtColor(img2_channel, cv2.COLOR_GRAY2BGR)
       
        # img1_channel_rgb = np.stack([img1_channel] * 3, axis=-1)
        # img2_channel_rgb = np.stack([img2_channel] * 3, axis=-1)

        superposed_img1 = cv2.addWeighted(heatmap_resize, alpha, img1_channel_rgb, 1 - alpha, 0)
        axs[0, idx].imshow(superposed_img1, cmap='jet')
        # axs[0, i].imshow(heatmap)
        axs[0, idx].axis('off')
        axs[0, idx].set_title(f'Heatmap - ' + mapList[i])

        superposed_img2 = cv2.addWeighted(heatmap_resize, alpha, img2_channel_rgb, 1 - alpha, 0)
        axs[1, idx].imshow(superposed_img2, cmap='jet')
        #axs[1, i].imshow(heatmap)
        axs[1, idx].axis('off')
        axs[1, idx].set_title(f'Heatmap - ' + mapList[i])

        idx = idx + 1
    
    fig.text(0.5, 0.95, img_info['session_1'], ha='center', va='center', fontsize=14, fontweight='bold')

    fig.text(0.5, 0.45, img_info['session_2'], ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.9])  # Ajustar para no sobreponer con los títulos
    
    # plt.show()

    #plt.tight_layout()

    # graphic to numpy array 
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) 

    plt.close(fig)
    return data

def save_gradcam_mean_images(img1, img2, heatmap, mapList, img_info, normalizer):
    img1, img2= normalizer.denormalize((img1, img2))
    alpha = 0.6
    colormap = cv2.COLORMAP_JET
    fig, axs = plt.subplots(2, img1.shape[0] *2, figsize=(16, 8))  # 2 filas (una por imagen), 4 columnas (una por canal)

    # Resize to have the same size as the image
    heatmap_resize = cv2.resize(heatmap, (img1.shape[2], img1.shape[1]))

    # Heatmap normalized between 0 and 255
    heatmap_resize = np.uint8(255 * heatmap_resize)

    heatmap_resize = cv2.applyColorMap(heatmap_resize, colormap)

    img1 = img1.permute(1, 2, 0).cpu().numpy()  # Change [C, H, W] to [H, W, C]
    img2 = img2.permute(1, 2, 0).cpu().numpy()  

    # plot original
    idx = 0

    for i in range(img1.shape[-1]):

        img1_channel = (img1[ :, :, i] * 255).astype(np.uint8)
        img2_channel = (img2[ :, :, i] * 255).astype(np.uint8)

        axs[0, idx].imshow(img1_channel, cmap='gray')
        axs[0, idx].axis('off')
        axs[0, idx].set_title(mapList[i])

        axs[1, idx].imshow(img2_channel, cmap='gray')
        #axs[1, i].imshow(heatmap)
        axs[1, idx].axis('off')
        axs[1, idx].set_title(mapList[i])

        idx = idx + 1

        img1_channel_rgb = cv2.cvtColor(img1_channel, cv2.COLOR_GRAY2BGR)
        img2_channel_rgb = cv2.cvtColor(img2_channel, cv2.COLOR_GRAY2BGR)
       
        # img1_channel_rgb = np.stack([img1_channel] * 3, axis=-1)
        # img2_channel_rgb = np.stack([img2_channel] * 3, axis=-1)

        superposed_img1 = cv2.addWeighted(heatmap_resize, alpha, img1_channel_rgb, 1 - alpha, 0)
        axs[0, idx].imshow(superposed_img1, cmap='jet')
        # axs[0, i].imshow(heatmap)
        axs[0, idx].axis('off')
        axs[0, idx].set_title(f'Heatmap - ' + mapList[i])

        superposed_img2 = cv2.addWeighted(heatmap_resize, alpha, img2_channel_rgb, 1 - alpha, 0)
        axs[1, idx].imshow(superposed_img2, cmap='jet')
        #axs[1, i].imshow(heatmap)
        axs[1, idx].axis('off')
        axs[1, idx].set_title(f'Heatmap - ' + mapList[i])

        idx = idx + 1
    
    fig.text(0.5, 0.95, img_info['session_1'], ha='center', va='center', fontsize=14, fontweight='bold')

    fig.text(0.5, 0.45, img_info['session_2'], ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.9])  # Ajustar para no sobreponer con los títulos

    #plt.tight_layout()

    # plt.show()

    # graphic to numpy array 
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) 

    plt.close(fig)
    return data

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # TRAIN
    eConfig = {
        'dir': 'default',
        'cfg_file':'conf.config1',
        'shift_da': -1,
        'angle_range': -1,
        'deactivate': -1,
        'cropEyePolar_px': -1,
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
    
    #Reading the config file
    cfg = importlib.import_module(eConfig['cfg_file'])

    # Create results directory
    results_dir = 'results/' +  eConfig['dir']

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if len(cfg.mapList) == 14: 
        if not os.path.exists(results_dir + '/' + 'allMaps') :
            os.makedirs(results_dir + '/'+ 'allMaps')
        results_path = results_dir + '/' + 'allMaps/'
    elif len(cfg.mapList) == 1:
        if not os.path.exists(results_dir + '/' + cfg.mapList[0]) :
            os.makedirs(results_dir + '/'+ cfg.mapList[0])
        results_path = results_dir + '/' + cfg.mapList[0] + '/'
    else:
        if not os.path.exists(results_dir + '/' + '_'.join(cfg.mapList)) :
            os.makedirs(results_dir + '/'+ '_'.join(cfg.mapList))
        results_path = results_dir + '/' + '_'.join(cfg.mapList) + '/'

    text_file = results_path + '/'+ eConfig['dir'] + '.txt' 


    rseed = int(eConfig['seed'])
    if rseed>=0:
        cfg.random_seed = rseed
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Reading conf from {}'.format(eConfig['cfg_file']))
    print(device)
    print('Random Seed: %d'%cfg.random_seed)

    with open(text_file, "w") as file:
        file.write('Reading conf from config --> {}'.format(eConfig['cfg_file']))
        file.write('Using {} device... \n' .format(device))
        file.write('Random seed = ' + eConfig['seed'])

    normalization_file = os.path.join(cfg.dataDir,'normalization.npz')
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
        
    if cropEyeBorder>=0:
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
        list_transforms = [RandomRotation(angle_range)] + list_transforms
    else:
        if hasattr(cfg, 'angle_range_da'):
            list_transforms = [RandomRotation(cfg.angle_range_da)] + list_transforms 
    
    if(float(eConfig["shift_da"]) != -1):
        shift_da = float(eConfig["shift_da"])
        list_transforms = [RandomTranslation(shift_da)] + list_transforms 
    else:
        if hasattr(cfg, 'shift_da'):
            list_transforms = [RandomTranslation(cfg.shift_da)] + list_transforms 
    
    if not hasattr(cfg, 'jitter_brightness'):
        cfg.jitter_brightness = 0
    if not hasattr(cfg, 'jitter_contrast'):
        cfg.jitter_contrast = 0

    if cfg.jitter_brightness>0 or cfg.jitter_contrast>0:
        list_transforms = [RandomJitter(cfg.jitter_brightness, cfg.jitter_contrast)] + list_transforms 
    
    if(float(eConfig["deactivate"]) != -1):
        deactivate = int(eConfig["deactivate"])
        list_transforms = list_transforms + [Deactivate(deactivate)]
        list_transforms_basic = list_transforms_basic + [Deactivate(deactivate)]
    else:
        if hasattr(cfg, 'deactivate'):
            list_transforms = list_transforms + [Deactivate(cfg.deactivate)]
            list_transforms_basic = list_transforms_basic + [Deactivate(cfg.deactivate)]
    
    if(float(eConfig["cropEyePolar_px"]) != -1):
        cropEyePolar_px= int(eConfig["cropEyePolar_px"])
        list_transforms = list_transforms + [cropEyePolar(cropEyePolar_px)]
        list_transforms_basic = list_transforms_basic + [cropEyePolar(cropEyePolar_px)]
    else:
        if hasattr(cfg, 'cropEyePolar_px'):
            list_transforms = list_transforms + [cropEyePolar(cfg.cropEyePolar_px)]
            list_transforms_basic = list_transforms_basic + [cropEyePolar(cfg.cropEyePolar_px)]
    
    transform_chain = transforms.Compose(list_transforms)
    transform_chain_basic = transforms.Compose(list_transforms_basic)

    # dataset
    full_dataset = PairEyeDataset(cfg.csvFile, cfg.imageDir,cfg.dataDir, cfg.error_BFSDir, cfg.error_PACDir, transform=transform_chain,mapList=cfg.mapList,test=False,random_seed=cfg.random_seed,testProp=0)
    random_generator = torch.Generator().manual_seed(rseed)
    dataset_sizes = len(full_dataset)*np.ones(cfg.numFolds) // float(cfg.numFolds)
    remSamples = len(full_dataset) - int(dataset_sizes.sum())
    for i in range(remSamples):
       dataset_sizes[i]+=1
    #Divide the dataset into folders
    fold_datasets = random_split(full_dataset,dataset_sizes.astype(int),generator=random_generator)
    
    total_scores = []
    total_scores_norm = []
    total_labels = []
    AUCs = np.zeros((cfg.numFolds,)) 
    best_auc_model = 0
    plot_image_best = []
    save_plot_best = []
    plot_batch_image_best = []
    save_plot_batch_best = []
    patients_val_batch = []
    
    for fold in range(cfg.numFolds):

        print("Fold ",fold)
        with open(text_file, "a") as file:
            file.write("Fold " + str(fold))
        
        # Set random seed for reproducibility
        random.seed(cfg.random_seed)
        npr.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)

        torch.backends.cudnn.benchmark = True

        if cfg.modelType!='Alexnet' and cfg.modelType!='VGG':
            torch.use_deterministic_algorithms(True,warn_only=True)
    
        val_idx = fold

        val_dataset = copy.deepcopy(fold_datasets[val_idx])
        val_dataset.dataset.transform = transform_chain_basic
        train_idx = np.ones((cfg.numFolds,),dtype=int)
        train_idx[val_idx] = 0
        train_dataset = [fold_datasets[idx] for idx in np.nonzero(train_idx)[0]]
        train_dataset = ConcatDataset(train_dataset)

        #Specify training dataset, with a batch size of 8, shuffle the samples, and parallelize with 4 workers
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_bs,
                        shuffle=True, num_workers=cfg.train_num_workers)
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
    
        #The loss is Contrastive
        criterion = ContrastiveLoss(margin = 0.4,alpha=1.0)
    
        # We will use SGD with momentum as optimizer
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=cfg.lr, momentum=cfg.momentum,weight_decay=cfg.wd)
        # optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=cfg.lr, alpha=0.99, eps=1e-08, weight_decay=cfg.wd,momentum=cfg.momentum);#, momentum=0,
        # optimizer_ft = optim.Adam(model_ft.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.wd);#, momentum=0,
        # Our scheduler starts with an lr=1e-3 and decreases by a factor of 0.1 every 7 epochs.
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=cfg.step_size, gamma=cfg.gamma)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,factor=cfg.gamma,patience=cfg.step_size)
        image_datasets = {'train' : train_dataset, 'val': val_dataset}
    
        dataloaders = {'train' : train_dataloader, 'val': val_dataloader}
              
        model_ft, best_auc_model, plot_image_best, save_plot_best, plot_batch_image_best, save_plot_batch_best, patients_val_batch = train_model(model_ft, image_datasets, cfg.mapList, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                               device, results_path, normalizer, num_epochs=cfg.num_epochs, max_epochs_no_improvement=cfg.max_epochs_no_improvement,min_epochs=5, batchsize_train = cfg.train_bs, batchsize_val = cfg.val_bs, best_auc_model=best_auc_model, grad_cam_status = cfg.grad_cam, plot_image_best=plot_image_best, save_plot_best=save_plot_best, plot_batch_image_best = plot_batch_image_best, save_plot_batch_best= save_plot_batch_best, patients_val_batch = patients_val_batch)
    
    if not os.path.exists(results_path + '/ssl_gradcam'):
            os.makedirs(results_path + '/ssl_gradcam')

    if not os.path.exists(results_path + '/ssl_gradcam_batch'):
            os.makedirs(results_path + '/ssl_gradcam_batch')
        
    for idx, image_data in enumerate(plot_image_best):
        plt.imshow(image_data)
        plt.title(save_plot_best[idx].replace('.png', ''))
        plt.axis('off')
        plt.savefig(results_path + '/ssl_gradcam/'+save_plot_best[idx], bbox_inches='tight', pad_inches=0.1)
        plt.clf()

    for idx, image_data in enumerate(plot_batch_image_best):
        plt.imshow(image_data)
        plt.title(save_plot_batch_best[idx].replace('.png', ''))
        plt.axis('off')
        plt.savefig(results_path + '/ssl_gradcam_batch/'+save_plot_batch_best[idx], bbox_inches='tight', pad_inches=0.1)
        plt.clf()

    print("Best AUC model saved", best_auc_model)
    with open(text_file, "a") as file:
            file.write("Best AUC model saved" + str(best_auc_model))

    with open(results_path + '/val_batch_patients.pkl', 'wb') as f:
        pickle.dump(patients_val_batch, f)
    
