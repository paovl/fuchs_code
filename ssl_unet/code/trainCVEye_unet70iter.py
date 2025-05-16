"""
Created on Mon Feb 10 11:49:10 2025
@author: pvltarife
"""
# imports
import torch
import torch.nn.functional as F
import os
import sys
import pandas as pd
import numpy as np
import pickle
import copy
import time
import importlib
import random
import warnings
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from models_unet import UNet
from EyeDataset_unet import EyeDataset
from dataAugmentation_unet import Normalize, ToTensor, cropEye, centerMask, centerMinPAC, reSize, PolarCoordinates, RandomRotation, RandomTranslation, RandomJitter, PatchHide
from MSELoss import MSELoss
from MAELoss import MAELoss
from NCCLoss import NCCLoss
from LocalNCCLoss import LocalNCCLoss
from SSIMLoss import SSIMLoss
from torchvision import transforms
import numpy.random as npr
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error 
import pdb
# from sklearn.metrics import r2_score
from SSIM_metric import SSIM_metric
from NCC_metric import NCC_metric
from LocalNCC_metric import LocalNCC_metric
from scipy.signal import correlate2d
from MSE_SSIM_Loss import MSESSIMLoss
from MSE_metric import MSE_metric
from MAE_metric import MAE_metric
from NCC_metric_map import NCC_metric_map

def r2_score(y_pred, y_true, mask= None, multioutput='mean'):
    """
    Calculate the R^2 score
    """
    n_outputs = y_true.shape[0]

    if mask is None:
        numerator = np.sum((y_true - y_pred) ** 2, axis = 0)
        denominator = np.sum((y_true - np.mean(y_true, axis=0))** 2, axis = 0)
        nonzero_numerator = numerator != 0
        nonzero_denominator = denominator != 0
        r2 = np.ones([n_outputs])
        valid_score = nonzero_denominator & nonzero_numerator
        r2[valid_score] = 1 - (
            numerator[valid_score] / denominator[valid_score]
        )
        r2[nonzero_numerator & ~nonzero_denominator] = 0.0

        if multioutput == 'raw_values':
            return r2
        else:
            r2 = np.mean(r2)
    else:
        y_sum_valid = np.sum(mask, axis = 1) # number of pixels valid for each sample

        numerator = np.sum((mask * (y_true - y_pred)) ** 2, axis = 1)

        y_true_sum  = np.sum(mask * y_true, axis = 1)

        y_true_mean = y_true_sum / y_sum_valid

        y_true_mean = y_true_mean[:, np.newaxis]
        denominator = np.sum((mask * (y_true - y_true_mean)) ** 2, axis=1)

        if multioutput == 'raw_values':
            r2 = np.full((n_outputs,), np.nan)
            denom_valid = denominator != 0
            r2[denom_valid] = 1 - (numerator[denom_valid] / denominator[denom_valid])
        else:
            denom_valid = denominator != 0
            r2 = np.ones([n_outputs])
            r2[denom_valid] = 1 - (numerator[denom_valid] / denominator[denom_valid])
            r2 = np.mean(r2)
    return r2

# Train model 
def train_model(model, image_datasets, mapList, dataloaders, criterion, optimizer, scheduler, device, results_path, normalizer, loss_name, num_epochs=25, max_epochs_no_improvement=10, min_epochs=10):
    
    since = time.time() # Start time

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    best_metric= - np.inf
    best_epoch = -1
    epochs_no_improvement = 0
    prev_lr = optimizer.param_groups[0]['lr']
    loss_values_train = []
    loss_values_val = []

    if loss_name == 'SSIM':
        ssim = SSIM_metric()
    elif loss_name == 'NCC':
        ncc = NCC_metric()
    elif loss_name == 'LocalNCC':
        local_ncc = LocalNCC_metric()
    elif loss_name == 'MSESSIM':
        ssim = SSIM_metric()
    
    #Loop of epochs
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        with open(text_file, "a") as file:
            file.write('\nEpoch {}/{}\n'.format(epoch, num_epochs - 1))
            file.write("-" * 10 + '\n')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set the model in training mode
            else:
                valset = True
                model.eval()   # Set the model in val mode (no grads)
                
            # Create variables to store outputs and labels
            outputs_m=[]
            ground_truth_m=[]
            outputs_masked_m = []
            ground_truth_masked_m = []
            mask_m = []
            mask_masked_m = []

            running_loss = 0.0
            mse_running_loss = 0.0
            ssim_running_loss = 0.0
            contSamples = 0

            # Iterate (loop of batches)
            nbatch = 0
            for img in dataloaders[phase]:

                batchsize = img['img'].shape[0]

                nbatch += 1

                # Image info
                img_info = {'idx': img['idx'],'patient':img['patient'], 'eye': img['eye'], 'session': img['session']}

                # Recover images 
                mask = img['mask'].to(device)
                mask = mask.permute(0, 3, 1, 2)
                label = img['label'].to(device)
                img = img['img'].to(device)

                # Set grads to zero
                optimizer.zero_grad()

                # Forward
                # Register ops only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(img) # Evaluate the model
                    # output = F.normalize(output,dim=0) NECESARIO?

                    # loss, output, ground_truth = criterion(output, mask, label, error, normalizer)
                    loss, output, ground_truth = criterion(output, mask, label, normalizer)
                    
                    # backward & parameters update only in train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Accumulate the running loss
                running_loss += loss.item() * batchsize

                # Store outputs and labels

                outputs_m.extend(output.detach().cpu())

                ground_truth_m.extend(ground_truth.cpu())

                # outputs_masked_m.extend(output[mask].detach().cpu().numpy())

                # outputs_masked_m.extend(output[mask].view(batchsize, -1).detach().cpu().numpy())

                # outputs_masked_m.extend((output * mask).sum(dim=1).view(batchsize, -1).detach().cpu().numpy())

                outputs_masked_m.extend((output * mask)[torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 0], torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 1], :, :].view(batchsize, -1).detach().cpu().numpy())
                
                # ground_truth_masked_m.extend(ground_truth[mask].cpu().numpy())
                
                # ground_truth_masked_m.extend(ground_truth[mask].view(batchsize, -1).detach().cpu().numpy())

                # ground_truth_masked_m.extend((ground_truth * mask).sum(dim=1).view(batchsize, -1).detach().cpu().numpy())

                ground_truth_masked_m.extend((ground_truth * mask)[torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 0], torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 1], :, :].view(batchsize, -1).detach().cpu().numpy())

                mask_masked_m.extend(mask[torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 0], torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 1], :, :].view(batchsize, -1).detach().cpu().numpy())

                # error_masked_m.extend(error[torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 0], torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 1], :, :].view(batchsize, -1).detach().cpu().numpy())

                mask_m.extend(mask.detach().cpu().int())

                # error_m.extend(error.detach().cpu())

                contSamples += batchsize
                            
            #Accumulated loss by epoch
            epoch_loss = running_loss / contSamples

            if loss_name == 'MSE' or loss_name == 'MAE':
                epoch_metric = r2_score(np.array(outputs_masked_m), np.array(ground_truth_masked_m), mask= np.array(mask_masked_m))

            elif loss_name == 'SSIM':
                epoch_metric, _, _ = ssim(torch.stack(outputs_m), torch.stack(mask_m).cpu(), torch.stack(ground_truth_m), normalizer)

            elif loss_name == 'NCC':
                epoch_metric, _, _ = ncc(torch.stack(outputs_m), torch.stack(mask_m), torch.stack(ground_truth_m), normalizer)
            
            elif loss_name == 'LocalNCC':
                epoch_metric, _, _ = local_ncc(torch.stack(outputs_m), torch.stack(mask_m), torch.stack(ground_truth_m), normalizer)
            
            elif loss_name == 'MSESSIM':
                epoch_r2 = r2_score(np.array(outputs_masked_m), np.array(ground_truth_masked_m), mask= np.array(mask_masked_m))
                epoch_ssim, _, _ = ssim(torch.stack(outputs_m), torch.stack(mask_m), torch.stack(ground_truth_m), normalizer)
                epoch_metric = criterion.alpha * epoch_r2 + (1 - criterion.alpha) * epoch_ssim

            r2_scores = r2_score(np.array(outputs_masked_m), np.array(ground_truth_masked_m), mask= np.array(mask_masked_m), multioutput='raw_values')

            # # epoch_ssim = epoch_metric
            if phase == 'train':
                loss_values_train.append(epoch_loss)
            else:
                loss_values_val.append(epoch_loss)

            if loss_name == 'MSE' or loss_name == 'MAE' or loss_name == 'MSESSIM':
                
                outputs_m = np.array(outputs_m, dtype='float')
                
                ground_truth_m = np.array(ground_truth_m, dtype='float')

            else:
                outputs_m = torch.stack(outputs_m).cpu().numpy()

                torch.stack(ground_truth_m).cpu().numpy()

            if phase == 'val':
            	
                scheduler.step(epoch_loss)
                if optimizer.param_groups[0]['lr']<prev_lr:
                    prev_lr=optimizer.param_groups[0]['lr']

            if loss_name == 'MSE' or loss_name == 'MAE':

                print('{} Loss: {:.4f}. R2 score: {:.4f}. lr: {} '.format(
                        phase, epoch_loss, epoch_metric, optimizer.param_groups[0]['lr']))
                
                with open(text_file, "a") as file:
                    file.write('{} Loss: {:.4f}. R2 score: {:.4f}. lr: {} \n'.format(
                        phase, epoch_loss, epoch_metric, optimizer.param_groups[0]['lr']))
            elif loss_name == 'SSIM':
                print('{} Loss: {:.4f}. SSIM index: {:.4f}. lr: {} '.format(
                        phase, epoch_loss, epoch_metric, optimizer.param_groups[0]['lr']))
                
                with open(text_file, "a") as file:
                    file.write('{} Loss: {:.4f}. SSIM index: {:.4f}. lr: {} \n'.format(
                        phase, epoch_loss, epoch_metric, optimizer.param_groups[0]['lr']))
            elif loss_name == 'NCC':
                print('{} Loss: {:.4f}. NCC: {:.4f}. lr: {} '.format(
                        phase, epoch_loss, epoch_metric, optimizer.param_groups[0]['lr']))
                
                with open(text_file, "a") as file:
                    file.write('{} Loss: {:.4f}. NCC: {:.4f}. lr: {} \n'.format(
                        phase, epoch_loss, epoch_metric, optimizer.param_groups[0]['lr']))
            elif loss_name == 'LocalNCC':
                print('{} Loss: {:.4f}. Local NCC: {:.4f}. lr: {} '.format(
                        phase, epoch_loss, epoch_metric, optimizer.param_groups[0]['lr']))
                
                with open(text_file, "a") as file:
                    file.write('{} Loss: {:.4f}. Local NCC: {:.4f}. lr: {} \n'.format(
                        phase, epoch_loss, epoch_metric, optimizer.param_groups[0]['lr']))
            
            elif loss_name == 'MSESSIM':
                print('{} Loss: {:.4f}. R2 score: {:.4f}. SSIM index: {:.4f}. Metric: {:.4f}. lr: {} '.format(
                        phase, epoch_loss, epoch_r2, epoch_ssim, epoch_metric, optimizer.param_groups[0]['lr']))
                
                with open(text_file, "a") as file:
                    file.write('{} Loss: {:.4f}. R2 score: {:.4f}. SSIM index: {:.4f}. Metric: {:.4f}. lr: {} \n'.format(
                        phase, epoch_loss, epoch_r2, epoch_ssim, epoch_metric, optimizer.param_groups[0]['lr']))
                
            # Deep copy of the best model
            if phase == 'val' and epoch >= min_epochs:
                if epoch_metric > best_metric and epoch_loss < best_loss:  # or (epoch_auc == 1.0 and epoch_loss < best_loss):
                    best_metric = epoch_metric
                    best_loss = epoch_loss # Save the best loss
                    best_model_wts = copy.deepcopy(model.state_dict()) # Save the best model
                    best_epoch = epoch # Save the best epoch
                    epochs_no_improvement = 0
                elif epoch_loss < best_loss:
                    epochs_no_improvement=0
                    model.load_state_dict(best_model_wts)
                else:
                    epochs_no_improvement+=1
                    model.load_state_dict(best_model_wts)
                        
        if epochs_no_improvement >= max_epochs_no_improvement:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best model in epoch {:02d} val R2 Score {:4f}'.format(best_epoch, best_metric))

    with open(text_file, "a") as file:
        file.write('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        file.write('Best model in epoch {:02d} val R2 Score {:4f}\n'.format(best_epoch, best_metric))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_metric

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # Config file
    eConfig = {
        'dir': 'default',
        'cfg_file':'conf.config_unet',
        'seed':'0',
        'dataset_seed':'0',
        'num': '1',
        'type': 'ring',
        'volume': 'False',
        'dilation': '0',
        'erosion': '0',
        'min_radius': '1.75',
        'min_arc': '90',
        'top_right_radius': '6',
        'loss':'SSIM',
        'alpha_loss':'0.5'
        }
    
    print(os.getenv("PYTORCH_CUDA_ALLOC_CONF"))

    # Parse command line arguments  
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
    
    weights_dir = results_path + 'weights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Create text file
    text_file = results_path + '/'+ eConfig['dir'] + '.txt'

    with open(text_file, "w") as file:
        file.write("eConfig\n")
        for key, value in eConfig.items():
            file.write('%s: %s\n' % (key, value))

    # Open normalization file
    normalization_file = os.path.join(cfg.db_path,'dataset_global_unet/data/normalization.npz')
    with open(normalization_file, 'rb') as f:
        normalization_data = pickle.load(f)
    
    #Filter using the map list
    idxCat = [normalization_data['categories'].index(cat) for cat in cfg.mapList]
    nMaps = len(cfg.mapList)

    normalizer = Normalize(mean=normalization_data['mean'][idxCat],std=normalization_data['std'][idxCat])

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Random seed
    rseed = int(eConfig['seed'])
    if rseed>=0:
        cfg.random_seed = rseed

    # Dataset seed
    dataset_seed = int(eConfig['dataset_seed'])
    if dataset_seed>=0:
        cfg.dataset_seed = dataset_seed

    # Random dataset generator
    rng = np.random.default_rng(seed=cfg.dataset_seed)

    print('Reading conf from {}'.format(eConfig['cfg_file']))
    print('Using {} device... \n' .format(device))
    print('Random Seed = %d'%cfg.random_seed)
    print('Random Dataset Seed = %d'%cfg.dataset_seed)

    with open(text_file, "a") as file:
        file.write('Reading conf from config --> {}'.format(eConfig['cfg_file']))
        file.write('Using {} device... \n' .format(device))
        file.write('Random seed = ' + eConfig['seed'])
        file.write('Random Dataset Seed = ' + eConfig['dataset_seed'])

    # Optional transformations

    # Dropout
    if not hasattr(cfg, 'dropout'):
        cfg.dropout = 0

    # Transformation pipeline 

    list_transforms = []

    # # Preprocessing

    # Crop Eye Border
    if hasattr(cfg, 'cropEyeBorder'):
        cropEyeBorder=cfg.cropEyeBorder
    else:
        cropEyeBorder=-1
    
    list_transforms=[cropEye(cfg.mapList,cropEyeBorder)] + list_transforms 
    
    # Center
    if hasattr(cfg, 'centerMethod'):
        centerMethod = cfg.centerMethod
    else:
        centerMethod = 0
    
    if centerMethod == 1:
        list_transforms = [centerMask(cfg.mapList)] + list_transforms 
    elif centerMethod == 2:
        list_transforms = [centerMinPAC(cfg.mapList)] + list_transforms 
    
    #  Resize
    list_transforms = list_transforms + [reSize(cfg.imSize)]

    # Polar coordinates
    if cfg.polar_coords:
        list_transforms = list_transforms + [PolarCoordinates()]

    # Patch Hiding
    if cfg.patch_hiding:
        list_transforms = list_transforms + [PatchHide(rng, patch_num = int(eConfig['num']), volume = (eConfig['volume'].lower() == 'true'), type = eConfig['type'], dilation = int(eConfig['dilation']), erosion = int(eConfig['erosion']), min_radius = float(eConfig['min_radius']), min_arc = float(eConfig['min_arc']), top_right_radius=float(eConfig['top_right_radius']))]

    # To tensor
    list_transforms = list_transforms + [ToTensor()]

    # Normalization
    list_transforms = list_transforms + [normalizer]

    list_transforms_basic = list_transforms # Basic pipeline

    # # Data augmentation

    # Random rotation
    if hasattr(cfg, 'angle_range_da'):
        list_transforms = [RandomRotation(cfg.angle_range_da, generator=rng)] + list_transforms 
    
    # Random translation
    if hasattr(cfg, 'shift_da'):
            list_transforms = [RandomTranslation(cfg.shift_da, generator=rng)] + list_transforms 
    
    # Random contrast and brigtness adjustment
    if not hasattr(cfg, 'jitter_brightness'):
        cfg.jitter_brightness = 0
    
    if not hasattr(cfg, 'jitter_contrast'):
        cfg.jitter_contrast = 0

    if cfg.jitter_brightness>0 or cfg.jitter_contrast>0:
        list_transforms = [RandomJitter(cfg.jitter_brightness, cfg.jitter_contrast, generator=rng)] + list_transforms 

    transform_chain_basic = transforms.Compose(list_transforms_basic)
    transform_chain = transforms.Compose(list_transforms)

    # Training iterations
    iterations = cfg.iterations

    for iter in np.arange(iterations):

        random_seed = int(iter * 10) + int(eConfig['seed'])

        # Dataset
        full_dataset = EyeDataset(cfg.csvFile, cfg.db_path, cfg.dataDir, transform = transform_chain, mapList = cfg.mapList)
        random_generator = torch.Generator().manual_seed(random_seed)
        dataset_sizes = len(full_dataset)*np.ones(cfg.numFolds) // float(cfg.numFolds)
        remSamples = len(full_dataset) - int(dataset_sizes.sum())
        for i in range(remSamples):
            dataset_sizes[i]+=1

        # Divide the dataset into folds
        fold_datasets = random_split(full_dataset, dataset_sizes.astype(int), generator = random_generator)

        print('\n\n' + '-'*5 + ' ITER %i ' %int(iter) + '-'*5 + '\n')
        with open(text_file, "a") as file:
            file.write('\n\n' + '-'*5 + ' ITER %i ' %int(iter) + '-'*5 + '\n\n')

        #Set random seed for reproducibility (weights initialization)
        random.seed(random_seed)
        npr.seed(random_seed)

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

        # Model
        model_ft = UNet(n_maps=nMaps, n_labels=1)
        
        model_ft = model_ft.to(device)

        # Early stopping 
        if not hasattr(cfg, 'max_epochs_no_improvement'):
            cfg.max_epochs_no_improvement = 10

        # MSE Loss
        if eConfig['loss'] == 'MSE':
            criterion = MSELoss()
        elif eConfig['loss'] == 'MAE':
            criterion = MAELoss()
        elif eConfig['loss'] == 'NCC':
            criterion = NCCLoss()
        elif eConfig['loss'] == 'LocalNCC':
            criterion = LocalNCCLoss()
        elif eConfig['loss'] == 'SSIM':
            criterion = SSIMLoss()
        elif eConfig['loss'] == 'MSESSIM':
            criterion = MSESSIMLoss(alpha=float(eConfig['alpha_loss']))

        # SGD with momentum optimizer
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=cfg.lr, momentum=cfg.momentum,weight_decay=cfg.wd)
        
        # Learning rate decay
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,factor=cfg.gamma,patience=cfg.step_size)

        image_datasets = {'train' : train_dataset, 'val': val_dataset}
        dataloaders = {'train' : train_dataloader, 'val': val_dataloader}
                
        model_ft, best_metric_model = train_model(model_ft, image_datasets, cfg.mapList, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                                device, results_path, normalizer, eConfig['loss'], num_epochs=cfg.num_epochs, max_epochs_no_improvement=cfg.max_epochs_no_improvement,min_epochs=5)
    
        if eConfig['loss'] == 'MSE' or eConfig['loss'] == 'MAE':
            print("Best R2 Score model saved: ", best_metric_model)
            with open(text_file, "a") as file:
                    file.write("Best R2 Score model saved: " + str(best_metric_model))
        elif eConfig['loss'] == 'SSIM':
            print("Best SSIM model saved: ", best_metric_model)
            with open(text_file, "a") as file:
                    file.write("Best SSIM model saved: " + str(best_metric_model))
        elif eConfig['loss'] == 'NCC':
            print("Best NCC model saved: ", best_metric_model)
            with open(text_file, "a") as file:
                    file.write("Best NCC model saved: " + str(best_metric_model))
        elif eConfig['loss'] == 'LocalNCC':
            print("Best Local NCC model saved: ", best_metric_model)
            with open(text_file, "a") as file:
                    file.write("Best Local NCC model saved: " + str(best_metric_model))
        elif eConfig['loss'] == 'MSESSIM':
            print("Best MSESSIM model saved: ", best_metric_model)
            with open(text_file, "a") as file:
                    file.write("Best MSESSIM model saved: " + str(best_metric_model))

        # Save model
        # torch.save(model_ft, results_path + "/model.pth")
        torch.save(model_ft.state_dict(), weights_dir + '/unet_self_supervised_iter' + str(iter) + '.pth')