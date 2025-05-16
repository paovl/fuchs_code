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
from sklearn.metrics import r2_score

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

    def __call__(self, input_tensor, mask):

        heatmap = torch.zeros(input_tensor.shape[0], self.activations.shape[2], self.activations.shape[3])
        
        output = self.model(input_tensor) # forward
        difference = torch.mean((input_tensor[mask] -  output[mask])**2)

        # Backward to compute gradients
        self.model.zero_grad() # reset gradients
        difference.backward(retain_graph=True) # compute gradients with respect to all parameters

        # Gradients' average
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3]) # Mean of last layer gradients (after batch norm) = Result one gradient for each channel

        # Weights
        for j in range(self.activations.shape[1]):
            self.activations[:, j, :, :] *= pooled_gradients[:,j].view(-1, 1, 1)

        # Heatmap
        heatmap = torch.mean(self.activations, dim=[1]).squeeze()
        heatmap = F.relu(heatmap)
        # heatmap = abs(heatmap)
        heatmap /= torch.max(heatmap)

        return heatmap.cpu().numpy()

# Train model 
def train_model(model, image_datasets, mapList, dataloaders, criterion, optimizer, scheduler, device, results_path, normalizer, num_epochs=25, max_epochs_no_improvement=10, min_epochs=10, best_r2_model = 0, grad_cam_status = False, plot_results = [], save_plot_results = [], save_plot_results_title = [], masked_imgs = [], output_imgs = [], imgs = [], residuals = [], patients_val_batch = []):
    
    since = time.time() # Start time

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    best_r2= - np.inf
    best_epoch = -1
    epochs_no_improvement = 0
    prev_lr = optimizer.param_groups[0]['lr']
    loss_values_train = []
    loss_values_val = []
    
    # Grad CAM
    target_layer = model.e4.conv
    # grad_cam = GradCAM(model, target_layer)
    
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
            running_loss = 0.0
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
                img_copy = img.clone().detach()
                label_copy = label.clone().detach()
                img_copy_mean = img.clone().detach()

                # Set grads to zero
                optimizer.zero_grad()

                # Forward
                # Register ops only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(img) # Evaluate the model
                    # output = F.normalize(output,dim=0) NECESARIO?
                    
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

                mask_m.extend(mask.detach().int())

                contSamples += batchsize
                            
            #Accumulated loss by epoch
            epoch_loss = running_loss / contSamples
            epoch_r2 = r2_score(np.array(outputs_masked_m), np.array(ground_truth_masked_m))

            r2_scores = r2_score(np.array(outputs_masked_m), np.array(ground_truth_masked_m), multioutput='raw_values')

            # epoch_r2 = mean_squared_error(np.array(outputs_masked_m), np.array(ground_truth_masked_m)).item()
            # outputs_ssim_m = (np.array(torch.clamp(normalizer.denormalize(torch.tensor(outputs_m)), 0, 1))[np.array(mask_m)]).reshape(-1, 1,  352, 112)
            # ground_truth_ssim_m = (np.array(torch.clamp(normalizer.denormalize(torch.tensor(ground_truth_m)), 0, 1))[np.array(mask_m)]).reshape(-1, 1,  352, 112)
            # epoch_ssim = ssim(outputs_ssim_m, ground_truth_ssim_m, gaussian_weights=True, channel_axis = 1, data_range = 1).item()

            # # epoch_ssim = epoch_r2
            if phase == 'train':
                loss_values_train.append(epoch_loss)
            else:
                loss_values_val.append(epoch_loss)
                
            outputs_m = np.array(outputs_m, dtype='float')
                
            ground_truth_m = np.array(ground_truth_m, dtype='float')

            if phase == 'val':
            	
                scheduler.step(epoch_loss)
                if optimizer.param_groups[0]['lr']<prev_lr:
                    prev_lr=optimizer.param_groups[0]['lr']

            print('{} Loss: {:.4f}. R2 score: {:.4f}. lr: {} '.format(
                phase, epoch_loss, epoch_r2, optimizer.param_groups[0]['lr']))
                
            with open(text_file, "a") as file:
                file.write('{} Loss: {:.4f}. R2 score: {:.4f}. lr: {} \n'.format(
                        phase, epoch_loss, epoch_r2, optimizer.param_groups[0]['lr']))
            
            # Deep copy of the best model
            if phase == 'val' and epoch >= min_epochs:
                if epoch_r2 > best_r2 and epoch_loss < best_loss:  # or (epoch_auc == 1.0 and epoch_loss < best_loss):
                    best_r2 = epoch_r2
                    best_loss = epoch_loss # Save the best loss
                    best_model_wts = copy.deepcopy(model.state_dict()) # Save the best model
                    best_epoch = epoch # Save the best epoch
                    epochs_no_improvement = 0
                    if best_r2 > best_r2_model:
                        torch.save(model.state_dict(), results_path + '/unet_self_supervised_fold.pth') # save the weights of the ResNet18
                        best_r2_model = best_r2

                        # Grad-CAM in validation
                        if grad_cam_status == True and phase == 'val':

                            plot_array_result = []
                            save_plot_result_title_array = []
                            save_plot_result_array = []
                            masked_imgs_array = []
                            output_imgs_array = []
                            imgs_array = []
                            residuals_array = []

                            # for each image
                            for i in np.arange(img.shape[0]):
                                plot_result, masked_img, output_img, img_original, title = save_output_images({'image': label_copy[i,:,:,:], 'mask': mask[i]}, output[i,:,:], mapList, {key: value[i] for key, value in img_info.items()}, r2_scores[i], normalizer)
                                plot_array_result.append(plot_result)
                                save_plot_result_array.append(img_info['patient'][i]+ '_' + str(img_info['eye'][i]) + '_' + img_info['session'][i]+ '.png')
                                save_plot_result_title_array.append(title)
                                masked_imgs_array.append(masked_img)
                                output_imgs_array.append(output_img)
                                imgs_array.append(img_original)
                        
                            residuals_aux = np.array(output_imgs_array) - np.array(imgs_array)
                            residuals_array = list(abs(residuals_aux))

                        patients_val_batch = {'idx': img_info['idx'], 'patient':img_info['patient'], 'r2_scores': r2_scores, 'output': outputs_m, 'ground_truth': ground_truth_m}
                        plot_results = plot_array_result
                        save_plot_results = save_plot_result_array
                        save_plot_results_title = save_plot_result_title_array
                        masked_imgs = masked_imgs_array
                        output_imgs = output_imgs_array
                        imgs = imgs_array
                        residuals = residuals_array

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
    print('Best model in epoch {:02d} val R2 Score {:4f}'.format(best_epoch, best_r2))

    with open(text_file, "a") as file:
        file.write('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        file.write('Best model in epoch {:02d} val R2 Score {:4f}\n'.format(best_epoch, best_r2))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_r2_model, plot_results, save_plot_results, save_plot_results_title, masked_imgs, output_imgs, imgs, residuals, patients_val_batch

def save_gradcam_images(img, heatmap, mapList, img_info, normalizer):
    img = normalizer.denormalize(img['image'])
    img = torch.clamp(img, 0, 1)
    alpha = 0.6
    colormap = cv2.COLORMAP_JET
    fig, axs = plt.subplots(1, img.shape[0] *2, figsize=(16, 8))  # 2 filas (una por imagen), 4 columnas (una por canal)

    # Resize to have the same size as the image
    heatmap_resize = cv2.resize(heatmap, (img.shape[2], img.shape[1]))

    # Heatmap normalized between 0 and 255
    heatmap_resize = np.uint8(255 * heatmap_resize)

    heatmap_resize = cv2.applyColorMap(heatmap_resize, colormap)

    img = img.permute(1, 2, 0).cpu().numpy()  # Change [C, H, W] to [H, W, C]

    # plot original
    idx = 0

    for i in range(img.shape[-1]):

        img_channel = (img[ :, :, i] * 255).astype(np.uint8)

        axs[idx].imshow(img_channel, cmap='gray')
        axs[idx].axis('off')
        axs[idx].set_title(mapList[i])

        idx = idx + 1

        img_channel_rgb= cv2.cvtColor(img_channel, cv2.COLOR_GRAY2BGR)

        superposed_img = cv2.addWeighted(heatmap_resize, alpha, img_channel_rgb, 1 - alpha, 0)
        axs[idx].imshow(superposed_img, cmap='jet')
        axs[idx].axis('off')
        axs[idx].set_title(f'Heatmap - ' + mapList[i])

        idx = idx + 1
    
    # plt.show()

    #plt.tight_layout()

    # graphic to numpy array 
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) 

    plt.close(fig)
    return data

def save_gradcam_mean_images(img, heatmap, mapList, img_info, normalizer):
    img = normalizer.denormalize(img['image'])
    img = torch.clamp(img, 0, 1)
    alpha = 0.6
    colormap = cv2.COLORMAP_JET
    fig, axs = plt.subplots(1, img.shape[0] *2, figsize=(16, 8))  # 2 filas (una por imagen), 4 columnas (una por canal)

    # Resize to have the same size as the image
    heatmap_resize = cv2.resize(heatmap, (img.shape[2], img.shape[1]))

    # Heatmap normalized between 0 and 255
    heatmap_resize = np.uint8(255 * heatmap_resize)

    heatmap_resize = cv2.applyColorMap(heatmap_resize, colormap)

    img = img.permute(1, 2, 0).cpu().numpy()  # Change [C, H, W] to [H, W, C]

    # plot original
    idx = 0

    for i in range(img.shape[-1]):

        img_channel = (img[ :, :, i] * 255).astype(np.uint8)

        axs[idx].imshow(img_channel, cmap='gray')
        axs[idx].axis('off')
        axs[idx].set_title(mapList[i])

        idx = idx + 1

        img_channel_rgb = cv2.cvtColor(img_channel, cv2.COLOR_GRAY2BGR)

        superposed_img = cv2.addWeighted(heatmap_resize, alpha, img_channel_rgb, 1 - alpha, 0)
        axs[idx].imshow(superposed_img, cmap='jet')
        axs[idx].axis('off')
        axs[idx].set_title(f'Heatmap - ' + mapList[i])
        idx = idx + 1

    #plt.tight_layout()

    # plt.show()

    # graphic to numpy array 
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) 

    plt.close(fig)
    return data

def save_output_images(img, output,mapList, img_info, r2_score, normalizer):
    mask = img['mask']

    img = normalizer.denormalize(img['image'])
    img = torch.clamp(img, 0, 1)
    output = normalizer.denormalize(output)
    output = torch.clamp(output, 0, 1)

    # plot

    selected_output = output * mask
    selected_img = img * mask

    map_idxs = torch.nonzero(mask.sum(dim=(1, 2)) > 0).squeeze().cpu().numpy()
    maps = np.array(mapList)[map_idxs]

    selected_output = selected_output[:, :, :]
    selected_img = selected_img[:, :, :]

    if len(selected_img.shape) < 3:
        selected_img = selected_img.unsqueeze(0)
        selected_output = selected_output.unsqueeze(0)

    # selected_output = selected_output.sum(dim=0, keepdim=True)
    # selected_img = selected_img.sum(dim=0, keepdim=True)

    # map_idx = torch.argmax(mask.sum(dim=(1,2))).detach().cpu().numpy()
    # map = mapList[map_idx]

    selected_output = selected_output.permute(1, 2, 0)
    selected_img = selected_img.permute(1, 2, 0)

    selected_residual = abs(selected_output - selected_img)

    selected_output = selected_output.cpu().numpy()
    selected_img = selected_img.cpu().numpy()
    selected_residual = selected_residual.cpu().numpy()

    # plot
    fig, axs = plt.subplots(selected_img.shape[2], 3, figsize=(25, 10))  # 2 filas (una por imagen), 4 columnas (una por canal)

    # complete
    mask = mask.permute(1, 2, 0)
    img = img.permute(1, 2, 0)
    output = output.permute(1, 2, 0)

    output_img = img.clone()
    output[~mask] = output_img[~mask]

    masked_img = img.clone()
    masked_img[mask] = 0

    img = img.cpu().numpy()  
    masked_img = masked_img.cpu().numpy()  
    output = output.cpu().numpy()

    if selected_img.shape[2] == 1:
        idx = 0

        selected_img_channel = (selected_img[:, : , 0] * 255).astype(np.uint8)

        axs[idx].imshow(selected_img_channel, cmap='gray')
        axs[idx].axis('off')
        axs[idx].set_title(f'Input - Map:')

        idx = idx + 1

        selected_output_channel = (selected_output[:, :, 0] * 255).astype(np.uint8)
        axs[idx].imshow(selected_output_channel, cmap='gray')
        axs[idx].axis('off')
        axs[idx].set_title(f'Output - Map:')

        idx = idx + 1

        selected_residual_channel = (selected_residual[:,:,0] * 255).astype(np.uint8)

        axs[idx].imshow(selected_residual_channel, cmap='gray')
        axs[idx].axis('off')
        axs[idx].set_title(f'Residual - Map:')
    else:
        for i in range(selected_img.shape[2]):

            idx = 0

            selected_img_channel = (selected_img[:, : , i] * 255).astype(np.uint8)

            axs[i, idx].imshow(selected_img_channel, cmap='gray')
            axs[i, idx].axis('off')
            axs[i, idx].set_title(f'Input - Map:')

            idx = idx + 1

            selected_output_channel = (selected_output[:, :, i] * 255).astype(np.uint8)
            axs[i, idx].imshow(selected_output_channel, cmap='gray')
            axs[i, idx].axis('off')
            axs[i, idx].set_title(f'Output - Map:')

            idx = idx + 1

            selected_residual_channel = (selected_residual[:,:,i] * 255).astype(np.uint8)

            axs[i, idx].imshow(selected_residual_channel, cmap='gray')
            axs[i, idx].axis('off')
            axs[i, idx].set_title(f'Residual - Map:')
    
    # plt.show()

    plt.tight_layout()

    title = f'Patient: {img_info["patient"]}, Eye: {img_info["eye"]}, Session: {img_info["session"]}, R2_score = {r2_score}'
    
    # graphic to numpy array 
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) 

    plt.close(fig)
    return data, masked_img, output, img, title

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # Config file
    eConfig = {
        'dir': 'default',
        'cfg_file':'conf.config_unet',
        'seed':'0',
        'dataset_seed':'0',
        'patch_size': '15',
        'patch_volume': 'True',
        'patch_type': 'ring',
        'patch_num':'3',
        'patch_w_start':'60',
        'patch_h_start': '352', 
        'dilation': '0',
        'loss':'SSIM'
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
        list_transforms = list_transforms + [PatchHide(int(eConfig['patch_size']), rng, volume = (eConfig['patch_volume'].lower() == 'true'), type = eConfig['patch_type'], patch_num = int(eConfig['patch_num']), dilation = int(eConfig['dilation']), w_start_limit = int(eConfig['patch_w_start']), h_start_limit = int(eConfig['patch_h_start']))]

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

    # Dataset
    full_dataset = EyeDataset(cfg.csvFile, cfg.db_path, cfg.dataDir, transform = transform_chain, mapList = cfg.mapList)
    dataset_sizes = len(full_dataset)*np.ones(cfg.numFolds) // float(cfg.numFolds)
    remSamples = len(full_dataset) - int(dataset_sizes.sum())
    for i in range(remSamples):
       dataset_sizes[i]+=1

    # Divide the dataset into folds
    random_generator = torch.Generator().manual_seed(cfg.dataset_seed)
    fold_datasets = random_split(full_dataset, dataset_sizes.astype(int), generator = random_generator)

    # Create val and train dataset
    val_idx = 0
    val_dataset = copy.deepcopy(fold_datasets[val_idx])
    val_dataset.dataset.transform = transform_chain_basic
    train_idx = np.ones((cfg.numFolds,),dtype=int)
    train_idx[val_idx] = 0
    train_dataset = [fold_datasets[idx] for idx in np.nonzero(train_idx)[0]]
    train_dataset = ConcatDataset(train_dataset)
    
    # Set random seed for reproducibility (weights initialization)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    npr.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Specify training dataset, with a batch size of 8, shuffle the samples, and parallelize with 4 workers
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_bs,
                    shuffle=True, num_workers=cfg.train_num_workers, generator= random_generator)
   
    #Validation dataset => No shuffle
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.val_bs,
                    shuffle=False, num_workers=cfg.val_num_workers, generator= random_generator)

    AUC = 0
    best_r2_model = - np.inf
    plot_results = []
    save_plot_results = []
    save_plot_results_title = []
    masked_imgs = []
    output_imgs = []
    imgs = []
    residuals = []
    patients_val_batch = {}

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

    # SGD with momentum optimizer
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=cfg.lr, momentum=cfg.momentum,weight_decay=cfg.wd)
    
    # Learning rate decay
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,factor=cfg.gamma,patience=cfg.step_size)

    image_datasets = {'train' : train_dataset, 'val': val_dataset}
    dataloaders = {'train' : train_dataloader, 'val': val_dataloader}
            
    model_ft, best_r2_model, plot_results, save_plot_results, save_plot_results_title, masked_imgs, output_imgs, imgs, residuals, patients_val_batch = train_model(model_ft, image_datasets, cfg.mapList, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                            device, results_path, normalizer, num_epochs=cfg.num_epochs, max_epochs_no_improvement=cfg.max_epochs_no_improvement,min_epochs=5, best_r2_model = best_r2_model, grad_cam_status = cfg.grad_cam, plot_results= plot_results, save_plot_results = save_plot_results, save_plot_results_title = save_plot_results_title, masked_imgs = masked_imgs, output_imgs= output_imgs, imgs = imgs, residuals = residuals, patients_val_batch = patients_val_batch)
   
    # Save gradcam images
    if not os.path.exists(results_path + '/ssl_gradcam'):
            os.makedirs(results_path + '/ssl_gradcam')

    if not os.path.exists(results_path + '/ssl_gradcam_batch'):
            os.makedirs(results_path + '/ssl_gradcam_batch')
    
    if not os.path.exists(results_path + '/ssl_results'):
            os.makedirs(results_path + '/ssl_results')

    for idx, image_data in enumerate(plot_results):
        plt.imshow(image_data)
        plt.title(save_plot_results_title[idx])
        plt.axis('off')
        plt.savefig(results_path + '/ssl_results/'+save_plot_results[idx], bbox_inches='tight', pad_inches=0.1)
        plt.clf()

    for idx in np.arange(len(masked_imgs)):
        for j, map in enumerate(cfg.mapList):
            cv2.imwrite(results_path + '/ssl_results/' + save_plot_results[idx].replace('.png', '_' + map +'_masked' +'.png'), np.uint16(65535*masked_imgs[idx][:,:,j]))
            cv2.imwrite(results_path + '/ssl_results/' + save_plot_results[idx].replace('.png', '_'+ map + '_output' +'.png'), np.uint16(65535*output_imgs[idx][:,:,j]))
            cv2.imwrite(results_path + '/ssl_results/' + save_plot_results[idx].replace('.png', '_'+ map +'_original' +'.png'), np.uint16(65535*imgs[idx][:,:,j]))
            cv2.imwrite(results_path + '/ssl_results/' + save_plot_results[idx].replace('.png', '_'+ map +'_residual' +'.png'), np.uint16(65535*residuals[idx][:,:,j]))
    print("Best R2 Score model saved: ", best_r2_model)
    with open(text_file, "a") as file:
        file.write("Best R2 Score model saved: " + str(best_r2_model))
  
    with open(results_path + '/val_batch_patients.pkl', 'wb') as f:
        pickle.dump(patients_val_batch, f)

    # Save model
    torch.save(model_ft, results_path + "/model.pth")
    torch.save(model_ft.state_dict(), results_path + "/model_weights.pth")