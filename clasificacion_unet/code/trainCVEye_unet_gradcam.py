
"""
    Created on Mon May 22 11:08:06 2023
    @author: igonzalez
"""

# Imports
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader,random_split,ConcatDataset
from EyeDataset_unet import EyeDataset
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
from dataAugmentation_unet import Normalize, ToTensor, RandomRotation,RandomTranslation,centerMask,centerMinPAC, cropEye, reSize, PolarCoordinates, centerCrop, RandomJitter, CartesianCoordinates
import time
import copy
# import config as cfg
from models_unet import UNetModel
from torchvision import transforms
import random
import numpy.random as npr
import sys
import importlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import warnings
warnings.filterwarnings("ignore")
import pdb

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for saving activations and gradients
        self.target_layer.register_forward_hook(self.save_activation) # capture activations in forward pass
        self.target_layer.register_full_backward_hook(self.save_gradient) # capture layer gradients in backward pass

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach() # grad from this

    def __call__(self, input_tensor, labels):

        # descongelar el gradiente
        # Descongelar bottleneck
        for param in self.model.b.parameters():
            param.requires_grad = True

        heatmap = torch.zeros(input_tensor.shape[0], self.activations.shape[2], self.activations.shape[3])

        for i in range(input_tensor.shape[0]):

            output = self.model(input_tensor[i, :, :, :].unsqueeze(0)) # forward
            
            self.model.zero_grad()
            output.backward(retain_graph=True) # backward pass to compute gradients
            
            # Gradients' average
            pooled_gradients = torch.mean(self.gradients, dim=[2, 3]) # Mean of last layer gradients (after batch norm) = Result one gradient for each channel
            # Weights
            for j in range(self.activations.shape[1]):
                self.activations[:, j, :, :] *= pooled_gradients[:,j].view(-1, 1, 1)

            # Heatmap
            heatmap[i, :, :] = torch.mean(self.activations, dim=[1]).squeeze()
            # heatmap = F.relu(heatmap)
            heatmap[i, :, :] = abs(heatmap[i, :, :])
            # heatmap /= torch.max(heatmap)
            heatmap[i, :, :] = (heatmap[i, :, :] - torch.min(heatmap[i, :, :])) / (torch.max(heatmap[i, :, :]) - torch.min(heatmap[i, :, :]))

        for param in self.model.b.parameters():
            param.requires_grad = False

        return heatmap.cpu().numpy()

# train_model parameters are the network (model), the criterion (loss),
# the optimizer, a learning scheduler (lr strategy), and the training epochs
def train_model(model, image_datasets, mapList, dataloaders,criterion, optimizer, scheduler, text_file, device, normalizer, num_epochs=25,max_epochs_no_improvement=10,min_epochs=10, plot_image_best = [], save_plot_best = [], original_images = [], gradcam_images= [], plt_save_gradcam_images = [],):
    
    since = time.time() # Time track for training session
    
    # Aux variables for best results
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0
    best_loss = np.inf
    best_epoch = -1
    epochs_no_improvement = 0
    prev_lr = optimizer.param_groups[0]['lr'] # Previous lr 
    target_layer = model.b
    grad_cam = GradCAM(model, target_layer)
    cartesian_coord = CartesianCoordinates()

    # Epoch loop
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
                model.eval()   # Set the model in val mode (no grads)

            """
            Freeze batchnorm
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    pdb.set_trace()
                    module=
                    pdb.set_trace()
                    track_running_stats = True
                    module.eval()
            """

            # Dataset size
            numSamples = len(image_datasets[phase])
            
            # Create variables to store outputs and labels
            outputs_m=np.zeros((numSamples,),dtype=float)
            labels_m=np.zeros((numSamples,),dtype=int)
            running_loss = 0.0
            contSamples=0 # Samples counter per batch
            
            # Iterate (batch loop)
            for sample in dataloaders[phase]:
                """
                For each train batch, the optimizer updates the weights by 
                back propagation of the loss and finding the new gradients. 

                With the validation batch there is no need to set the gradients,
                because we only want to see how the CNNs works with an alternative 
                set of data.
                """
                inputs_info = sample
                inputs = sample['image'].to(device).float()
                inputs_copy = inputs.clone().detach()
                inputs_copy_mean = inputs.clone().detach()
                bsizes = {'bsize_x': sample['bsize_x'], 'bsize_y': sample['bsize_y']}
                labels = sample['label'].to(device).float()

                # plt.figure(1);plt.hist(inputs[:,0,:,:].flatten().cpu().numpy(),100);
                # plt.figure(2);plt.hist(inputs[:,1,:,:].flatten().cpu().numpy(),100);
                # plt.figure(3);plt.hist(inputs[:,2,:,:].flatten().cpu().numpy(),100);
                # plt.show()
                
                #Batch Size
                batchSize = labels.shape[0]
                
                # Set grads to zero
                optimizer.zero_grad()

                # Forward
                # Register outputs only in train
                with torch.set_grad_enabled(phase == 'train'):
                     
                    outputs = model(inputs)
    
                    loss = criterion(outputs, labels) # Obtain the loss 
                    
                    # Backward & parameters update (only in train)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Accumulate the running loss
                running_loss += loss.item() * inputs.size(0)

                #Apply a softmax to the output
                outputs= outputs.data
                # Store outputs and labels 
                outputs_m [contSamples:contSamples+batchSize]= outputs.cpu().numpy()
                labels_m [contSamples:contSamples+batchSize]= labels.cpu().numpy()
                contSamples+=batchSize

            # Accumulated loss by epoch
            epoch_loss = running_loss / numSamples   
            # Compute the AUCs at the end of the epoch
            auc = roc_auc_score(labels_m, outputs_m) 
            
            #At the end of an epoch, update the lr scheduler    
            if phase == 'val':
                # scheduler.step()
                scheduler.step(epoch_loss)
                if optimizer.param_groups[0]['lr']<prev_lr:
                    prev_lr=optimizer.param_groups[0]['lr']
                    
            # if phase == 'val':
                # print([labels_m,outputs_m])
            #And the Average AUC
            epoch_auc = auc

            print('{} Loss: {:.4f} AUC: {:.4f} lr: {}'.format(
                phase, epoch_loss, auc,optimizer.param_groups[0]['lr']))
            with open(text_file, "a") as file:
                file.write('{} Loss: {:.4f} AUC: {:.4f} lr: {} \n'.format(
                    phase, epoch_loss, auc,optimizer.param_groups[0]['lr']))

            # Deep copy of the best model
            if phase == 'val' and epoch >= min_epochs:# Min number of epochs to train the model
                if epoch_auc > best_auc or (epoch_auc == 1.0 and epoch_loss<best_loss): #and epoch_loss<best_loss: 
                    best_auc = epoch_auc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch # Best epoch, with the highest improvement
                    epochs_no_improvement = 0 # Counter returns to 0
                    
                    plot_array = []
                    save_plot_array = []
                    original_images_array = []
                    gradcam_images_array = []
                    gradcam_mean_images_array = []
                    save_gradcam_images_array = []

                    heatmaps = grad_cam(inputs, labels)

                    heatmap_mean = np.mean(heatmaps, axis=0)  # Para batch

                    for i in np.arange(inputs.shape[0]):
                        plot_image, plt_img, plt_gradcam_img = save_gradcam_images(inputs_copy[i,:,:], heatmaps[i,:,:], mapList, {key: value[i] for key, value in inputs_info.items()}, normalizer, cartesian_coord)
                        plot_array.append(plot_image)
                        # si encuentro hryc o multicentrico en img path
                        # if 'hryc' in inputs_info['img_path'][i]:
                        #     db = 'hryc'
                        # else:
                        #     db = 'multi'
                        save_plot_array.append(inputs_info['id'][i] +'.png')
                        original_images_array.append(plt_img)
                        gradcam_images_array.append(plt_gradcam_img)
                        save_gradcam_images_array.append(inputs_info['id'][i]+ '_'+'.png')

                    for i in np.arange(inputs.shape[0]):
                        plot_batch_image, _, plt_gradcam_mean_img = save_gradcam_mean_images(inputs_copy_mean[i,:,:], heatmap_mean, mapList, {key: value[i] for key, value in inputs_info.items()}, normalizer, cartesian_coord)
                        gradcam_mean_images_array.append(plt_gradcam_mean_img)

                    plot_image_best = plot_array
                    save_plot_best = save_plot_array
                    original_images = original_images_array
                    gradcam_images = gradcam_images_array
                    gradcam_mean_images = gradcam_mean_images_array
                    plt_save_gradcam_images = save_gradcam_images_array

                elif epoch_loss<best_loss: # Counter returns to 0
                    epochs_no_improvement = 0
                    model.load_state_dict(best_model_wts)
                else:
                    epochs_no_improvement += 1 # There is no improvement 
                    model.load_state_dict(best_model_wts)

        if epochs_no_improvement >= max_epochs_no_improvement: 
            # No improvement --> finish train
            break

    time_elapsed = time.time() - since # Stop the time lapse 
    
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best model in epoch {:02d} val AUC {:4f}'.format(best_epoch,best_auc))
    with open(text_file, "a") as file:
        file.write('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        file.write('Best model in epoch {:02d} val AUC {:4f}\n'.format(best_epoch,best_auc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, plot_image_best, save_plot_best, original_images, gradcam_images, gradcam_mean_images, plt_save_gradcam_images

def test_model(model, image_dataset, dataloader,device):
    
    since = time.time() # Time track for training session
    model.eval()   # Set the model in eval mode (no grads)

    # Dataset size
    numSamples = len(image_dataset)
    
    # Create variables to store outputs and labels
    outputs_m=np.zeros((numSamples,),dtype = float)
    labels_m=np.zeros((numSamples,),dtype = int)
    ids_m = []
    running_loss = 0.0
    
    contSamples = 0 # Samples counter per batch
    
    # Iterate (batch loop)
    for sample in dataloader:
        # Gets the samples and labels per batch
        ids = sample['id']
        bsizes = {'bsize_x': sample['bsize_x'], 'bsize_y': sample['bsize_y']}
        inputs = sample['image'].to(device).float()
        labels = sample['label'].to(device).float()
        # Batch Size
        batchSize = labels.shape[0]

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
               
            #outputs= torch.mean(outputs, axis = 1)
            # Store ids, outputs, labels 
            ids_m [contSamples:contSamples+batchSize] = ids
            outputs_m [contSamples:contSamples+batchSize]= outputs.cpu().numpy()
            labels_m [contSamples:contSamples+batchSize] = labels.cpu().numpy()
            contSamples += batchSize
            
    return ids_m, outputs_m, labels_m


def save_gradcam_images(img, heatmap, mapList, img_info, normalizer, cartesian_coord):
    img = normalizer.denormalize(img)
    alpha = 0.6
    colormap = cv2.COLORMAP_JET
    fig, axs = plt.subplots(1, img.shape[0] *2, figsize=(16, 8))  # 2 filas (una por imagen), 4 columnas (una por canal)

    # Resize to have the same size as the image
    heatmap_resize_img = cv2.resize(heatmap, (img.shape[2], img.shape[1]))

    # Heatmap normalized between 0 and 255
    heatmap_resize_img = np.uint8(255 * heatmap_resize_img)

    heatmap_resize_img = cv2.applyColorMap(heatmap_resize_img, colormap)

    heatmap_resize_img_bgr = cv2.cvtColor(heatmap_resize_img, cv2.COLOR_RGB2BGR)

    heatmap_resize_img_bgr_cartesian = cartesian_coord(heatmap_resize_img_bgr,)
    
    img = img.permute(1, 2, 0).cpu().numpy()  # Change [C, H, W] to [H, W, C])

    img_bw = np.zeros_like(img)
    img_mask = img > 0
    img_bw[img_mask] = 0.65

    img_cartesian = cartesian_coord(img)
    # pdb.set_trace()
       
    # plot original
    idx = 0

    channels_superposed_img = np.zeros((img.shape[0], img.shape[1], 3, img.shape[2]))

    channels_superposed_img_bw = np.zeros((img_bw.shape[0], img_bw.shape[1], 3, img_bw.shape[2]))

    for i in range(img.shape[-1]):

        img_channel = (img[ :, :, i] * 255).astype(np.uint8)
        img_bw_channel = (img_bw[ :, :, i] * 255).astype(np.uint8)

        axs[idx].imshow(img_channel, cmap='gray')
        axs[idx].axis('off')
        axs[idx].set_title(mapList[i])

        idx = idx + 1

        img_channel_rgb= cv2.cvtColor(img_channel, cv2.COLOR_GRAY2RGB)
        img_channel_bgr = cv2.cvtColor(img_channel, cv2.COLOR_GRAY2BGR)
        img_bw_channel_bgr = cv2.cvtColor(img_bw_channel, cv2.COLOR_GRAY2BGR)

        superposed_img = cv2.addWeighted(heatmap_resize_img, alpha, img_channel_rgb, 1 - alpha, 0)
        superposed_img_bgr = cv2.addWeighted(heatmap_resize_img_bgr, alpha, img_channel_bgr, 1 - alpha, 0)
        superposed_img_bw_bgr = cv2.addWeighted(heatmap_resize_img_bgr, alpha, img_bw_channel_bgr, 1 - alpha, 0)
        
        axs[idx].imshow(superposed_img)
        axs[idx].axis('off')
        axs[idx].set_title(f'Heatmap - ' + mapList[i])
        idx = idx + 1

        channels_superposed_img[:, :, :, i] = superposed_img_bgr
        channels_superposed_img_bw[:, :, :, i] = superposed_img_bw_bgr

    channels_superposed_img_cartesian = np.zeros((img_cartesian.shape[0], img_cartesian.shape[1], 3, img_cartesian.shape[2]))
    channels_superposed_img_cartesian_bw = np.zeros((img_cartesian.shape[0], img_cartesian.shape[1], 3, img_cartesian.shape[2]))

    for i in range(channels_superposed_img.shape[3]):
        channels_superposed_img_cartesian[:, :, :, i] = cartesian_coord(channels_superposed_img[:, :, :, i])
        channels_superposed_img_cartesian_bw[:, :, :, i] = cartesian_coord(channels_superposed_img_bw[:, :, :, i])
        
    plt.tight_layout()
    # plt.show()

    #plt.tight_layout()

    # graphic to numpy array 
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) 

    plt.close(fig)
    return data, img_cartesian, channels_superposed_img_cartesian_bw

def save_gradcam_mean_images(img, heatmap, mapList, img_info, normalizer, cartesian_coord):
    img = normalizer.denormalize(img)
    alpha = 0.6
    colormap = cv2.COLORMAP_JET
    fig, axs = plt.subplots(1, img.shape[0] *2, figsize=(16, 8))  # 2 filas (una por imagen), 4 columnas (una por canal)

    # Resize to have the same size as the image
    heatmap_resize_img = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
    
    # Heatmap normalized between 0 and 255
    heatmap_resize_img = np.uint8(255 * heatmap_resize_img)

    heatmap_resize_img = cv2.applyColorMap(heatmap_resize_img, colormap)

    heatmap_resize_img_bgr = cv2.cvtColor(heatmap_resize_img, cv2.COLOR_RGB2BGR)

    heatmap_resize_img_bgr_cartesian = cartesian_coord(heatmap_resize_img_bgr,)
    
    img = img.permute(1, 2, 0).cpu().numpy()  # Change [C, H, W] to [H, W, C]

    img_bw = np.zeros_like(img)
    img_mask = img > 0
    img_bw[img_mask] = 0.65

    img_cartesian = cartesian_coord(img)

    channels_superposed_img = np.zeros((img.shape[0], img.shape[1], 3, img.shape[2]))

    channels_superposed_img_bw = np.zeros((img_bw.shape[0], img_bw.shape[1], 3, img_bw.shape[2]))

    # plot original
    idx = 0

    for i in range(img.shape[-1]):

        img_channel = (img[ :, :, i] * 255).astype(np.uint8)
        img_bw_channel = (img_bw[ :, :, i] * 255).astype(np.uint8)

        axs[idx].imshow(img_channel, cmap='gray')
        axs[idx].axis('off')
        axs[idx].set_title(mapList[i])

        idx = idx + 1

        img_channel_rgb = cv2.cvtColor(img_channel, cv2.COLOR_GRAY2RGB)
        img_channel_bgr = cv2.cvtColor(img_channel, cv2.COLOR_GRAY2BGR)
        img_bw_channel_bgr = cv2.cvtColor(img_bw_channel, cv2.COLOR_GRAY2BGR)

        superposed_img = cv2.addWeighted(heatmap_resize_img, alpha, img_channel_rgb, 1 - alpha, 0)
        superposed_img_bgr = cv2.addWeighted(heatmap_resize_img_bgr, alpha, img_channel_bgr, 1 - alpha, 0)
        superposed_img_bw_bgr = cv2.addWeighted(heatmap_resize_img_bgr, alpha, img_bw_channel_bgr, 1 - alpha, 0)
        
        axs[idx].imshow(superposed_img, cmap='jet')
        axs[idx].axis('off')
        axs[idx].set_title(f'Heatmap - ' + mapList[i])

        idx = idx + 1

        channels_superposed_img[:, :, :, i] = superposed_img_bgr
        channels_superposed_img_bw[:, :, :, i] = superposed_img_bw_bgr
    
    channels_superposed_img_cartesian = np.zeros((img_cartesian.shape[0], img_cartesian.shape[1], 3, img_cartesian.shape[2]))
    channels_superposed_img_cartesian_bw = np.zeros((img_cartesian.shape[0], img_cartesian.shape[1], 3, img_cartesian.shape[2]))

    for i in range(channels_superposed_img.shape[3]):
        channels_superposed_img_cartesian[:, :, :, i] = cartesian_coord(channels_superposed_img[:, :, :, i])
        channels_superposed_img_cartesian_bw[:, :, :, i] = cartesian_coord(channels_superposed_img_bw[:, :, :, i])
        
    plt.tight_layout()  # Ajustar para no sobreponer con los títulos

    # plt.show()

    # graphic to numpy array 
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) 

    plt.close(fig)
    return data, img, channels_superposed_img_cartesian_bw

if __name__ == '__main__':

    """ Arguments
        cfg_file: configuration file that contains CNN and training info
        db: database 'HRYC', 'multi' or 'global'
        bio: 
            - 0: full database
            - 1: biomarkers database
        dir: directory where data directories are saved
        data_dir: directory where data files are saved
        results_dir: directory where results are saved
        seed: random seed
        type: CNN architecture configuration
            - 0: baseline (avg pool)
            - 1: baseline (no pool)
            - 2: modification (polar pool)
        fusion: only used if type == 2
            0: no fusion, only modified branch (polar pool)
            1: fusion 1 with baseline
            2: fusion 2 with baseline
            3: fusion 3 with baseline
            4: fusion 4 with baseline
            5: fusion 5 with baseline
        rings: array with different rings radius
        arcs: array with different arcs radius
        max_radius: max radius in the input image (in mm)
        max_angle: max angle in the input image (in degrees)
    """
    eConfig = {
        'cfg_file' : 'conf.config_70iter',
        'db': 'global',
        'bio':'0',
        'dir': 'SEP',
        'data_dir': 'ransac_TH_1.5_r_45',
        'results_dir': 'medical_rings3_angles3_fusion5_50epochs',
        'seed' : '0',
        'dir_weights_file':'none',
        'weights_seed': '-1',
        'weights_dseed': '-1',
        'max_angle': '360',
        'weights_ssl_file': 'unet_self_supervised_fold.pth'
        }

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
    
    # Reading config file
    cfg = importlib.import_module(eConfig['cfg_file'])
    
    eConfig['max_radius_px'] = cfg.imSize[0] / 2 # Max radius in px

    # Build csv files 
    db_path = "../datasets/dataset_" + eConfig['db']
    dir_file = eConfig['dir']
    if eConfig['bio'] == 1:
        csvFile='../datasets/' + 'dataset_' + eConfig['db'] + '/' + eConfig['dir'] + '/' + eConfig['data_dir'] + '/annotation_biomarkers.csv'
    else:
        csvFile='../datasets/' + 'dataset_' + eConfig['db'] + '/' + eConfig['dir'] + '/' + eConfig['data_dir'] + '/annotation.csv'
    imageDir = db_path + '/'+ dir_file + '/' + eConfig['data_dir'] + '/data'

    # Create results directory
    results_dir = 'results/' +  eConfig['dir']
    if len(cfg.mapList) == 14: 
        results_path = results_dir + '/' + 'allMaps' + '/'+ eConfig['results_dir']
    elif len(cfg.mapList) == 1:
        results_path = results_dir + '/' + cfg.mapList[0] + '/'+ eConfig['results_dir']
    else:
        results_path = results_dir + '/' + '_'.join(cfg.mapList) + '/'+ eConfig['results_dir']
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if len(cfg.mapList) == 14: 
        if not os.path.exists(results_dir + '/' + 'allMaps') :
            os.makedirs(results_dir + '/'+ 'allMaps')
    elif len(cfg.mapList) == 1:
        if not os.path.exists(results_dir + '/' + cfg.mapList[0]) :
            os.makedirs(results_dir + '/'+ cfg.mapList[0])
    else:
        if not os.path.exists(results_dir + '/' + '_'.join(cfg.mapList)) :
            os.makedirs(results_dir + '/'+ '_'.join(cfg.mapList))

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    #only to test SSL

    results_path = results_path + '/' + eConfig['dir_weights_file'] + "_gradcam"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    model_weights_path = results_path + '/model_weights'
    if not os.path.exists(model_weights_path):
        os.makedirs(model_weights_path)

    text_file = results_path + '/'+ eConfig['results_dir'] + '.txt' 

    # Read ssl model
    path_weights_file = '../../ssl_unet/code/results/'

    if (eConfig['dir_weights_file'] != 'none') and ((eConfig['weights_seed'] != '-1') and (eConfig['weights_dseed'] != '-1')):
        if len(cfg.mapList) == 14: 
            model_ssl = torch.load(path_weights_file + eConfig['dir_weights_file']+ '/allMaps/' + eConfig['weights_ssl_file'])
        elif len(cfg.mapList) == 1:
            model_ssl = torch.load(path_weights_file + eConfig['dir_weights_file']+ '/' + cfg.mapList[0] + '/' + eConfig['weights_ssl_file'])
        else:
            model_ssl = torch.load(path_weights_file + eConfig['dir_weights_file']+ '/' + '_'.join(cfg.mapList) + '/' + eConfig['weights_ssl_file'])
    else:
        model_ssl = None
    
    #model.load_state_dict(checkpoint)
    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print('Reading conf from config --> {}'.format(eConfig['cfg_file']))
    print('Using {} device...' .format(device))
    print('Random seed = ' + eConfig['seed'])
    with open(text_file, "w") as file:
        file.write('Reading conf from config --> {}'.format(eConfig['cfg_file']))
        file.write('Using {} device... \n' .format(device))
        file.write('Random seed = ' + eConfig['seed'])

    # Get normalization data 
    normalization_file = os.path.join(imageDir,'normalization.npz')
    with open(normalization_file, 'rb') as f:
        normalization_data = pickle.load(f) # Recover the normalization data and transform it to a python object
    
    # Filter normalization data using the map list from the config file
    idxCat = [normalization_data['categories'].index(cat) for cat in cfg.mapList]
    nMaps = len(cfg.mapList)

    # Checks if there is any dropout 
    if not hasattr(cfg, 'dropout'):
        cfg.dropout=0
    
    #   First transformations
    #   Resize to 224 x 224 pixels
    #   Transform into polar coordinates 
    #   Tranform into tensor
    #   Normalize every sample with the normalization data, in order to obtain mean=0 and std=0
    normalizer = Normalize(mean=normalization_data['mean'][idxCat],std=normalization_data['std'][idxCat])
    
    if cfg.polar_coords:
        list_transforms=[reSize(cfg.imSize),
                         PolarCoordinates(eConfig['max_radius_px'], float(eConfig['max_angle'])),
                         ToTensor(),
                         normalizer]
    else:
        list_transforms=[reSize(cfg.imSize),
                         ToTensor(),
                         normalizer]
    
    # Crop image border
    if hasattr(cfg, 'cropEyeBorder'):
        cropEyeBorder = cfg.cropEyeBorder
    else:
        cropEyeBorder = -1

    list_transforms = [cropEye(cfg.mapList,cropEyeBorder)] + list_transforms 
    
    # Center image
    if hasattr(cfg, 'centerMethod'):
        centerMethod = cfg.centerMethod
    else:
        centerMethod=0
           
    # Apply center method
    if centerMethod == 1:
        list_transforms = [centerMask(cfg.mapList)] + list_transforms 
    elif centerMethod == 2:
        list_transforms = [centerMinPAC(cfg.mapList)] + list_transforms 
    
    # Combine all transformations in a basic chain
    transform_chain_basic = transforms.Compose(list_transforms)

    # Data augmentation
    # Geometric: Random rotation and random translation
    if hasattr(cfg, 'angle_range_da'):
        list_transforms = [RandomRotation(cfg.angle_range_da)] + list_transforms 
    
    if hasattr(cfg, 'shift_da'):
        list_transforms = [RandomTranslation(cfg.shift_da)] + list_transforms 
    
    # Photometric: Random ajustment of brightness and constrast
    if not hasattr(cfg, 'jitter_brightness'):
        cfg.jitter_brightness = 0

    if not hasattr(cfg, 'jitter_contrast'):
        cfg.jitter_contrast = 0

    if cfg.jitter_brightness > 0 or cfg.jitter_contrast > 0:
        list_transforms = [RandomJitter(cfg.jitter_brightness, cfg.jitter_contrast)] + list_transforms 
    
    # Combine all transformations for data augmentation
    transform_chain = transforms.Compose(list_transforms)

    # Training iterations
    # iterations = cfg.iterations
    iterations = 1

    # Result variables 
    total_scores=[]
    total1_scores=[]
    total2_scores=[]
    total_scores_norm=[]
    total1_scores_norm=[]
    total2_scores_norm=[]
    total_labels=[]
    total_ids = []
    AUCs = np.zeros((iterations,))
    AUCs_out1 = np.zeros((iterations,))
    AUCs_out2 = np.zeros((iterations,))

    plot_image_best = []
    save_plot_best = []
    original_images = []
    gradcam_images = []
    plt_save_gradcam_images = []

    full_dataset = EyeDataset(csvFile, imageDir, transform = transform_chain, mapList = cfg.mapList, test = False, random_seed = int(eConfig['seed']), testProp = 0)
    
    all_idxs = full_dataset.dataset['img_id'].index.astype(int)
    all_ids = full_dataset.dataset['img_id'].values
    all_labels = full_dataset.dataset['label'].values.astype(int)

    scores_matrix = np.full((len(full_dataset), iterations), np.nan)
    scores1_matrix = np.full((len(full_dataset), iterations), np.nan)
    scores2_matrix = np.full((len(full_dataset), iterations), np.nan)

    # iterations loop
    for iter in np.arange(iterations):

        # Random seed
        random_seed = int(iter * 10) + int(eConfig['seed'])

        # Spli dataset in five folds
        full_dataset = EyeDataset(csvFile, imageDir, transform = transform_chain, mapList = cfg.mapList, test = False, random_seed = random_seed, testProp = 0)
        random_generator = torch.Generator().manual_seed(random_seed)
        dataset_sizes = len(full_dataset)*np.ones(cfg.numFolds)//float(cfg.numFolds) # Fold size
        remSamples = len(full_dataset)-int(dataset_sizes.sum()) # Remaining samples
        for i in range(remSamples):
            dataset_sizes[i] += 1
        fold_datasets = random_split(full_dataset, dataset_sizes.astype(int), generator = random_generator)

        print('\n\n' + '-'*5 + ' ITER %i ' %int(iter) + '-'*5 + '\n')
        with open(text_file, "a") as file:
            file.write('\n\n' + '-'*5 + ' ITER %i ' %int(iter) + '-'*5 + '\n\n')

        # Set random seed for reproducibility
        random.seed(random_seed)
        npr.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Asegura determinismo

        # Use deterministic algorithms if the model type is one of the below
        if cfg.modelType != 'Alexnet' and cfg.modelType != 'VGG':
            torch.use_deterministic_algorithms(True, warn_only = True) 
        
        # Test dataset
        test_idx = random.randint(0, cfg.numFolds - 1)
        test_dataset = copy.deepcopy(fold_datasets[test_idx])
        test_dataset.dataset.transform = transform_chain_basic # Basic transformations and NOT data augmentation
        test_idxs = test_dataset.indices

        # Validation dataset
        val_idx = test_idx + 1
        if val_idx == cfg.numFolds:
            val_idx = 0
        val_dataset = copy.deepcopy(fold_datasets[val_idx])
        val_dataset.dataset.transform = transform_chain_basic # Basic transformations and NOT data augmentation
        
        # Train dataset
        train_idx = np.ones((cfg.numFolds,),dtype=int) # Train fold number
        train_idx[test_idx] = 0
        train_idx[val_idx] = 0
        train_dataset = [fold_datasets[idx] for idx in np.nonzero(train_idx)[0]]
        train_dataset = ConcatDataset(train_dataset) # Concat all the remaining folds

        # Train dataloader => Suffle
        train_dataloader = DataLoader(train_dataset, batch_size = cfg.train_bs,
                        shuffle = False, num_workers = cfg.train_num_workers)
        
        # Validation dataloader => No shuffle
        val_dataloader = DataLoader(val_dataset, batch_size = cfg.val_bs,
                        shuffle = False, num_workers = cfg.val_num_workers)
        
        # Test dataloader => No shuffle
        test_dataloader = DataLoader(test_dataset, batch_size = cfg.val_bs,
                        shuffle = False, num_workers = cfg.val_num_workers)
        
        # Create an instance for the CNN model type
        model_ft = UNetModel(nMaps, 1, model_ssl, freeze=True)
        
        # Max number of epochs with no improvement 
        if not hasattr(cfg, 'max_epochs_no_improvement'):
            cfg.max_epochs_no_improvement = 10
        
        model_ft = model_ft.to(device)
        criterion = nn.BCEWithLogitsLoss()# Cross-entropy loss
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=cfg.lr, momentum=cfg.momentum,weight_decay=cfg.wd) # SGD with momentum

        # Other optimizer options 
        # optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=cfg.lr, alpha=0.99, eps=1e-08, weight_decay=cfg.wd,momentum=cfg.momentum);#, momentum=0, 
        # optimizer_ft = optim.Adam(model_ft.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.wd);#, momentum=0, 
        # Our scheduler starts with an lr=1e-3 and decreases by a factor of 0.1 every 7 epochs.
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=cfg.step_size, gamma=cfg.gamma)

        # This scheduler reduces the learning rate. Note = Sometimes its better to start with a high leraning rate and then reduce it
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,factor=cfg.gamma,patience=cfg.step_size)

        # Dictionaries with the two train and validation datasets and dataloaders
        image_datasets = {'train' : train_dataset, 'val': val_dataset}
        dataloaders = {'train' : train_dataloader, 'val': val_dataloader}

        # Train model    
        model_ft, plot_image_best, save_plot_best, original_images, gradcam_images, gradcam_mean_images, plt_save_gradcam_images = train_model(model_ft, image_datasets, cfg.mapList, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, text_file,
                               device, normalizer, num_epochs=cfg.num_epochs, max_epochs_no_improvement=cfg.max_epochs_no_improvement,min_epochs=10)

        # Test model
        test_ids, test_scores, test_labels = test_model(model_ft, test_dataset, test_dataloader, device)
        
        # Saves the total scores and labels obtained for every test fold 
        # ids and labels by iterations
        total_ids.append(test_ids)
        total_labels.append(test_labels)

        # Logits by iteration
        total_scores.append(test_scores)
        
        # AUCs
        AUCs[iter] = roc_auc_score(test_labels.flatten(), test_scores.flatten())

        # Logits matriz for each id
        scores_matrix[test_idxs, iter] = test_scores

        print('testAUC %f \n'%(AUCs[iter]))
        
        with open(text_file, "a") as file:
            file.write('testAUC %f\n'%(AUCs[iter]))
            
        # Norm scores
        test_scores = test_scores - test_scores.min()
        total_scores_norm.append(test_scores/ test_scores.max()) # Test scores normalizated

        # save model weights 
        # torch.save(model_ft.state_dict(), model_weights_path + '/model_weights_iter' + str(iter) + '.pth')

        if not os.path.exists(results_path + '/ssl_gradcam'):
            os.makedirs(results_path + '/ssl_gradcam')

        if not os.path.exists(results_path + '/gradcam'):
                os.makedirs(results_path + '/gradcam')
        
        if not os.path.exists(results_path + '/gradcam_mean'):
                os.makedirs(results_path + '/gradcam_mean')

        if not os.path.exists(results_path + '/original'):
                os.makedirs(results_path + '/original')
        
        for idx, image_data in enumerate(plot_image_best):
            plt.imshow(image_data)
            plt.title(save_plot_best[idx].replace('.png', ''))
            plt.axis('off')
            plt.savefig(results_path + '/ssl_gradcam/'+save_plot_best[idx], bbox_inches='tight', pad_inches=0.1)
            plt.clf()

        for idx in np.arange(len(original_images)):
            for j, map in enumerate(cfg.mapList):
                cv2.imwrite(results_path + '/original/' + plt_save_gradcam_images[idx].replace('.png', '_' + map +'.png'), np.uint16(65535*original_images[idx][:,:,j]))
                cv2.imwrite(results_path + '/gradcam/' + plt_save_gradcam_images[idx].replace('.png', '_'+ map +'.png'), gradcam_images[idx][:,:, :,j])
                cv2.imwrite(results_path + '/gradcam_mean/' + plt_save_gradcam_images[idx].replace('.png', '_'+ map +'.png'), gradcam_mean_images[idx][:,:,:,j])
        
    # Show results
    print('mean testAUC accross iterations: %f \n'%(np.mean(AUCs)))

    with open(text_file, "a") as file:
            # Escribe en el file
            file.write('\n mean testAUC accross iterations: %f \n'%(np.mean(AUCs)))

    # Save results
    save_results_path = results_path + '/results_nn'
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    
    # Logits by iteration
    with open(save_results_path + '/out_scores.pkl', 'wb') as f:
        pickle.dump(total_scores, f)
    
    # DICT --> Norm scores by iteration

    dict_scores = {'img_id': total_ids, 'score': total_scores_norm, 'label': total_labels}
    with open(save_results_path +'/dict_scores.pkl', "wb") as f:
        pickle.dump(dict_scores, f)

    # DICT --> Logits matrix

    dict_raw_scores = {'img_id': all_ids, 'score': scores_matrix, 'label': all_labels}
    with open(save_results_path +'/dict_raw_scores.pkl', "wb") as f:
        pickle.dump(dict_raw_scores, f)
    
    # AUC
    
    with open(save_results_path +'/AUCs.pkl', "wb") as f:
        pickle.dump(AUCs, f)
        