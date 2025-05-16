# Imports
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader,random_split,ConcatDataset
from EyeDataset_unet_duplicate import EyeDataset
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
from dataAugmentation_unet_duplicate import Normalize, ToTensor, RandomRotation,RandomTranslation,centerMask,centerMinPAC, cropEye, reSize, PolarCoordinates, centerCrop, RandomJitter, PatchHide_duplicate
import time
import copy
# import config as cfg
from models_unet_duplicate import UNet
from torchvision import transforms
import random
import numpy.random as npr
import sys
import importlib
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import cv2

def test_model(model, image_dataset, dataloader, normalizer, device):
    
    since = time.time() # Time track for training session
    model.eval()   # Set the model in eval mode (no grads)

    # Dataset size
    numSamples = len(image_dataset)
    
    # Create variables to store outputs and labels
    outputs_m = []
    ground_truth_m = []
    ids_m = []
    paths_m = []
    mask_m = []
    
    contSamples = 0 # Samples counter per batch
    
    # # Iterate (batch loop)
    for sample in dataloader:
        # Gets the samples and labels per batch
        ids = sample['img_id']
        paths = sample['img_path']

        mask = sample['mask']
        mask = mask.permute(0, 1, 4, 2, 3)
        ground_truth = sample['label']
        input = sample['img']

        # fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        # mask_selected = mask[44, 1, :, :, :]
        # axes[0].imshow(mask_selected[0, :, :].cpu().numpy(), cmap='gray')
        # axes[1].imshow(mask_selected[1, :, :].cpu().numpy(), cmap='gray')
        # axes[2].imshow(mask_selected[2, :, :].cpu().numpy(), cmap='gray')
        # plt.show()

        c = input.shape[1]

        output = torch.zeros(input.shape[0], input.shape[2], input.shape[3], input.shape[4])
        output_ground_truth =  torch.zeros(input.shape[0], input.shape[2], input.shape[3], input.shape[4])
        output_mask = torch.zeros(input.shape[0], input.shape[2], input.shape[3], input.shape[4])

        for j in range(c):
            input_model = input[:, j, :, :, :]
            mask_model = mask[:, j, :, :, :]
            ground_truth_model = ground_truth[:, j, :, :, :]

            output_model = model(input_model.to(device))
            output_model = output_model.detach().cpu()

            ground_truth_model_map = ground_truth_model[:, j, :, :]
            output_model_map = output_model[:, j, :, :]
            mask_model_map = mask_model[:, j, :, :]
            output_model_map[~mask_model_map] = ground_truth_model_map[~mask_model_map]

            # plot every map 
            # fig, axes = plt.subplots(2, nMaps, figsize=(15, 5))
            # for j in range(nMaps):
            #     axes[0, j].imshow(img_ground_truth[j, :, :], cmap='gray', vmin=0, vmax=1)
            #     axes[0, j].set_title(f'Original Map {j+1}')
            #     axes[0, j].axis('off')

            #     axes[1, j].imshow(img_output[j, :, :], cmap='gray', vmin=0, vmax=1)
            #     axes[1, j].set_title(f'Augmented Map {j+1}')
            #     axes[1, j].axis('off')
            # plt.tight_layout()
            # plt.savefig(img_image_dupe_path + ".png", dpi=300)
            # plt.close(fig)

            output[:, j, :, :] = output_model_map
            output_ground_truth[:, j, :, :] = ground_truth_model_map
            output_mask[:, j, :, :] = mask_model_map
        
        ids_m.extend(ids)
        paths_m.extend(paths)
        outputs_m.append(output)
        ground_truth_m.append(output_ground_truth)
        mask_m.append(output_mask)
    
    outputs_m = torch.cat(outputs_m, dim=0)
    ground_truth_m = torch.cat(ground_truth_m, dim=0)
    mask_m = torch.cat(mask_m, dim=0)

    return ids_m, paths_m, outputs_m, ground_truth_m, mask_m

if __name__ == '__main__':
    eConfig = {
        'cfg_file' : 'conf.config_70iter_duplicate',
        'db': 'global',
        'bio':'0',
        'dir': 'SEP',
        'data_dir': 'ransac_TH_1.5_r_45',
        'results_dir': 'medical_rings3_angles3_fusion5_50epochs',
        'seed' : '0',
        'dataset_seed' : '0',
        'dir_weights_file':'unet_map_seed0_dseed0_num1_MSELoss(mask_dilation3)',
        'max_angle': '360',
        'weights_ssl_file': 'unet_self_supervised_fold.pth',
        'dilation': '3',
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

    results_path = results_path + '/' + eConfig['dir_weights_file']

    # Read ssl model weights
    path_weights_file = '../../ssl_unet/code/results/'

    if (eConfig['dir_weights_file'] != 'none'):
        if len(cfg.mapList) == 14: 
            model_ssl_weights = torch.load(path_weights_file + eConfig['dir_weights_file']+ '/allMaps/' + eConfig['weights_ssl_file'])
        elif len(cfg.mapList) == 1:
            model_ssl_weights = torch.load(path_weights_file + eConfig['dir_weights_file']+ '/' + cfg.mapList[0] + '/' + eConfig['weights_ssl_file'])
        else:
            model_ssl_weights = torch.load(path_weights_file + eConfig['dir_weights_file']+ '/' + '_'.join(cfg.mapList) + '/' + eConfig['weights_ssl_file'])
    else:
        model_ssl_weights = None
    
    #model.load_state_dict(checkpoint)
    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get normalization data 
    normalization_file = os.path.join(imageDir,'normalization.npz')
    with open(normalization_file, 'rb') as f:
        normalization_data = pickle.load(f) # Recover the normalization data and transform it to a python object
    
    # Filter normalization data using the map list from the config file
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

    list_transforms = list_transforms + [PatchHide_duplicate(rng, dilation = int(eConfig['dilation']))]

    # To tensor
    list_transforms = list_transforms + [ToTensor()]

    # Normalization
    list_transforms = list_transforms + [normalizer]

    transform_chain = transforms.Compose(list_transforms)

    test_dataset = EyeDataset(csvFile, imageDir, transform = transform_chain, mapList = cfg.mapList, test = False, random_seed = rseed, testProp = 0)

    # Set random seed for reproducibility (weights initialization)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    npr.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_dataloader = DataLoader(test_dataset, batch_size=cfg.val_bs, shuffle=False, num_workers=cfg.val_num_workers, pin_memory=True)

    model_ssl = UNet(n_maps=len(cfg.mapList), n_labels=1)

    # Load weights
    model_ssl.load_state_dict(model_ssl_weights)
    model_ssl.to(device)

    test_ids, test_paths, test_output, test_ground_truth, test_mask = test_model(model_ssl, test_dataset, test_dataloader, normalizer, device)

    # Guardar resultados

    df = pd.read_csv(csvFile)

    dbs = [imageDir.replace('dataset_global', 'dataset_hryc'), imageDir.replace('dataset_global', 'dataset_multicentrico')]

    for db in dbs: 
        data_dupe_path = db.rsplit('/data', 1)[0] + '/data_dupe'
        data_dupe_ground_truth_path = db.rsplit('/data', 1)[0] + '/ground_truth_dupe'
        data_dupe_image_path = db.rsplit('/data', 1)[0] + '/images_dupe'
        data_dupe_image_ground_truth_path = db.rsplit('/data', 1)[0] + '/images_ground_truth_dupe'

        if not os.path.exists(data_dupe_path):
            os.makedirs(data_dupe_path)
        if not os.path.exists(data_dupe_image_path):
            os.makedirs(data_dupe_image_path)
        if not os.path.exists(data_dupe_ground_truth_path):
            os.makedirs(data_dupe_ground_truth_path)
        if not os.path.exists(data_dupe_image_ground_truth_path):
            os.makedirs(data_dupe_image_ground_truth_path)

    # Guardamos datos

    for i in range(len(test_ids)):
        # Read the image
        img_path = test_paths[i]
        img_dupe_path = img_path.replace("/data/", "/data_dupe/")
        img_ground_truth_path = img_dupe_path.replace("/data_dupe/", "/ground_truth_dupe/")
        img_image_dupe_path = img_dupe_path.replace("/data_dupe/", "/images_dupe/")
        img_image_ground_truth_path = img_dupe_path.replace("/data_dupe/", "/images_ground_truth_dupe/")
        img_id = test_ids[i]
        img_output = test_output[i]
        img_ground_truth = test_ground_truth[i]
        img_mask = test_mask[i]

        img_output_denormalized = img_output.clone()
        img_ground_truth_denormalized = img_ground_truth.clone()

        img_output_denormalized = normalizer.denormalize(img_output_denormalized)
        img_output_denormalized = torch.clamp(img_output_denormalized, 0, 1)
        img_ground_truth_denormalized = normalizer.denormalize(img_ground_truth_denormalized)
        img_ground_truth_denormalized = torch.clamp(img_ground_truth_denormalized, 0, 1)

        for c in range(len(cfg.mapList)):
            img_output_map = img_output[c, :, :]
            img_ground_truth_map = img_ground_truth[c, :, :]
            img_output_denormalized_map = img_output_denormalized[c, :, :]
            img_ground_truth_denormalized_map = img_ground_truth_denormalized[c, :, :]

            # Save the image
            img_output_file = '%s_%s.npy'%(img_dupe_path, cfg.mapList[c])
            np.save(img_output_file, img_output_map)

            img_ground_truth_file = '%s_%s.npy'%(img_ground_truth_path, cfg.mapList[c])
            np.save(img_ground_truth_file, img_ground_truth_map)

            # Save images

            img_image_output_file = '%s_%s.png'%(img_image_dupe_path, cfg.mapList[c])
            cv2.imwrite(img_image_output_file, np.uint16(65535*img_output_denormalized_map.detach().numpy()))

            img_image_ground_truth_file = '%s_%s.png'%(img_image_ground_truth_path, cfg.mapList[c])
            cv2.imwrite(img_image_ground_truth_file, np.uint16(65535*img_ground_truth_denormalized_map.detach().numpy()))
        
        img_output = normalizer.denormalize(img_output)
        img_output = torch.clamp(img_output, 0, 1)
        img_ground_truth = normalizer.denormalize(img_ground_truth)
        img_ground_truth = torch.clamp(img_ground_truth, 0, 1)

        img_output = img_output.detach().cpu().numpy()
        img_ground_truth = img_ground_truth.detach().cpu().numpy()

        # plot every map 
        fig, axes = plt.subplots(2, nMaps, figsize=(15, 5))
        for j in range(nMaps):
            axes[0, j].imshow(img_ground_truth[j, :, :], cmap='gray', vmin=0, vmax=1)
            axes[0, j].set_title(f'Original Map {j+1}')
            axes[0, j].axis('off')

            axes[1, j].imshow(img_output[j, :, :], cmap='gray', vmin=0, vmax=1)
            axes[1, j].set_title(f'Augmented Map {j+1}')
            axes[1, j].axis('off')
        plt.tight_layout()
        plt.savefig(img_image_dupe_path + ".png", dpi=300)
        plt.close(fig)






    
  