
"""
    Created on Mon May 22 11:08:06 2023
    @author: igonzalez
"""

# Imports
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
from PIL import Image
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader,random_split,ConcatDataset, Subset
from torchvision import transforms, utils, models
import pandas as pd
from skimage import io, transform, color, morphology, util
from EyeDataset import EyeDataset
from sklearn.metrics import roc_auc_score
import numpy as np
import cv2
import pdb
import pickle
from dataAugmentation import Normalize, ToTensor, RandomRotation,RandomTranslation,centerMask,centerMinPAC, cropEye, reSize, PolarCoordinates, centerCrop, RandomJitter
import time
import copy
# import config as cfg
from models_modified import Resnet18Model,Resnet34Model,Resnet50Model,InceptionModel,AlexnetModel,VGGModel,Mobilev3large
from models_modified import RegnetY400mModel,RegnetY800mModel,RegnetY3_2gModel,RegnetY32gModel
import random
import numpy.random as npr
import sys
import importlib
import matplotlib.pyplot as plt


# Train_model parameters are the network (model), the criterion (loss),
# the optimizer, a learning scheduler (lr strategy), and the training epochs
def train_model(model, image_datasets, dataloaders,criterion, optimizer, scheduler, text_file, device, type, num_epochs=25,max_epochs_no_improvement=10,min_epochs=10):
    
    since = time.time() # Time track for training session
    
    # Aux variables for best results
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0
    best_loss = np.inf
    best_epoch = -1
    epochs_no_improvement = 0
    prev_lr=optimizer.param_groups[0]['lr'] # Previous lr 

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
                model.eval()   # Set the model in eval mode (no grads)

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
            
            # Create variables to store outputs and labels for each phase individually
            outputs_m = np.zeros((numSamples,),dtype=float)
            out1_m = np.zeros((numSamples,),dtype=float)
            out2_m = np.zeros((numSamples,),dtype=float)
            labels_m = np.zeros((numSamples,),dtype=int)
            running_loss = 0.0
            contSamples = 0 # Samples counter per batch
            
            # Iterate (batch loop)
            for sample in dataloaders[phase]:
                """
                For each train batch, the optimizer update the weights by 
                back propagation of the loss and finding the new gradients. 

                With the validation batch there is no need to set the gradients,
                because we only want to see how the CNNs works with an alternative 
                set of data. 
                """ 
                inputs = sample['image'].to(device).float()
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
                    if type != 2: 
                        outputs = model(inputs)[:,0]
                        out1 = torch.zeros(outputs.shape).to(device)
                        out2 = torch.zeros(outputs.shape).to(device)
                    else:
                        out1, out2, outputs = model(inputs) # Evaluate the model
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
                outputs_m [contSamples:contSamples+batchSize] = outputs.cpu().numpy()
                out1_m[contSamples:contSamples+batchSize] = out1.cpu().detach().numpy()
                out2_m[contSamples:contSamples+batchSize] = out2.cpu().detach().numpy()
                labels_m [contSamples:contSamples+batchSize]= labels.cpu().numpy()
                contSamples+=batchSize

            #Accumulated loss by epoch
            epoch_loss = running_loss / numSamples   
            #Compute the AUCs at the end of the epoch
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
            if phase == 'val' and epoch>=min_epochs:# Min number of epochs to train the model
                if epoch_auc > best_auc or (epoch_auc == 1.0 and epoch_loss < best_loss): #and epoch_loss<best_loss: 
                    best_auc = epoch_auc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch # Best epoch, with the highest improvement
                    epochs_no_improvement = 0 # Counter returns to 0
                elif epoch_loss<best_loss: # Counter returns to 0
                    epochs_no_improvement = 0
                    model.load_state_dict(best_model_wts)
                else:
                    epochs_no_improvement += 1 # There is no improvement 
                    model.load_state_dict(best_model_wts)

        if epochs_no_improvement >= max_epochs_no_improvement: 
            # No improvement --> Training finishes
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
    return model

def test_model(model, image_dataset, dataloader,device, type):
    
    since = time.time() # Time track for training session
    model.eval()   # Set the model in val mode (no grads)

    # Dataset size
    numSamples = len(image_dataset)
    
    # Create variables to store outputs and labels
    outputs_m=np.zeros((numSamples,),dtype = float)
    out1_m = np.zeros((numSamples,),dtype=float)
    out2_m = np.zeros((numSamples,),dtype=float)
    labels_m=np.zeros((numSamples,),dtype = int)
    ids_m = []
    running_loss = 0.0
    
    contSamples=0 # Samples counter per batch
    
    # Iterate (batch loop)
    for sample in dataloader:
        # Gets the samples and labels per batch
        ids = sample['id']
        inputs = sample['image'].to(device).float()
        labels = sample['label'].to(device).float()
        # Batch Size
        batchSize = labels.shape[0]

        with torch.set_grad_enabled(False):
            if type !=2: 
                outputs = model(inputs)[:,0]
                out1 = torch.zeros(outputs.shape).to(device)
                out2 = torch.zeros(outputs.shape).to(device)
            else:
                out1, out2, outputs = model(inputs) # Evaluate the model
            #outputs= torch.mean(outputs, axis = 1)
            # Store ids, outputs, labels 
            ids_m [contSamples:contSamples+batchSize] = ids
            outputs_m [contSamples:contSamples+batchSize]= outputs.cpu().numpy()
            out1_m[contSamples:contSamples+batchSize] = out1.cpu().detach().numpy()
            out2_m[contSamples:contSamples+batchSize] = out2.cpu().detach().numpy()
            labels_m [contSamples:contSamples+batchSize] = labels.cpu().numpy()
            contSamples += batchSize
            
    return ids_m, outputs_m, out1_m, out2_m, labels_m

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
        'dir': 'RESNET',
        'data_dir': 'ransac_TH_1.5_r_45',
        'results_dir': 'medical_rings3_angles3_fusion5_hospital',
        'seed' : '0', 
        'type':'2',
        'delete_last_layer': '1',
        'fusion':'5',
        'max_radius': '7',
        'max_angle': '360', 
        'bio':'0'
        }
    eConfig['rings'] = [1.0,3.0,5.0]
    eConfig['arcs'] = [120.0, 240.0, 360.0]
    
    args = sys.argv[1::]
    i = 0
    while i < len(args):
        key = args[i]
        if key == 'rings' or key == 'angles':
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

    # Build csv files path
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

    text_file = results_path + '/'+ eConfig['results_dir'] + '.txt'

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
    
    # First transformations
    #   Resize to 224 x 224 pixels
    #   Transform into polar coordinates 
    #   Tranform into tensor
    #   Normalize every sample with the normalization data, in order to obtain mean=0 and std=0
    
    if cfg.polar_coords:
        list_transforms=[reSize(cfg.imSize),
                         PolarCoordinates(eConfig['max_radius_px'], float(eConfig['max_angle'])),
                         ToTensor(),
                         Normalize(mean=normalization_data['mean'][idxCat],std=normalization_data['std'][idxCat])]
    else:
        list_transforms=[reSize(cfg.imSize),
                         ToTensor(),
                         Normalize(mean=normalization_data['mean'][idxCat],std=normalization_data['std'][idxCat])]
    
    # Crop image border
    if hasattr(cfg, 'cropEyeBorder'):
        cropEyeBorder = cfg.cropEyeBorder
    else:
        cropEyeBorder = -1

    if cropEyeBorder >= 0:
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

    # Result variables 
    total_scores=[]
    total1_scores=[]
    total2_scores=[]
    total_scores_norm=[]
    total1_scores_norm=[]
    total2_scores_norm=[]
    total_labels=[]
    total_ids = []
    AUCs = np.zeros((cfg.numFolds,))
    AUCs_out1 = np.zeros((cfg.numFolds,))
    AUCs_out2 = np.zeros((cfg.numFolds,))

    full_dataset = EyeDataset(csvFile, imageDir, transform = transform_chain, mapList = cfg.mapList, test = False, random_seed = eConfig['seed'], testProp = 0)
        
    all_idxs = full_dataset.dataset['img_id'].index.astype(int)
    all_ids = full_dataset.dataset['img_id'].values
    all_labels = full_dataset.dataset['label'].values.astype(int)

    scores_matrix = np.full((len(full_dataset), cfg.numFolds), np.nan)
    scores1_matrix = np.full((len(full_dataset), cfg.numFolds), np.nan)
    scores2_matrix = np.full((len(full_dataset), cfg.numFolds), np.nan)

    # Get hospitals names
    hospitals_names = full_dataset.dataset['hospital'].unique()
    fold_datasets = []
    for hospital in hospitals_names: 
        idxs = np.where(full_dataset.dataset['hospital'] == hospital)[0].tolist()
        fold_dataset = Subset(full_dataset, idxs)
        fold_datasets.append(fold_dataset)
    
    # each loop --> different hospital info for test dataset
    for i in np.arange(cfg.numFolds):

        # Random seed
        random_generator = torch.Generator().manual_seed(int(eConfig['seed']))

        # Set random seed for reproducibility
        random.seed(int(eConfig['seed']))
        npr.seed(int(eConfig['seed']))
        torch.manual_seed(int(eConfig['seed']))

        print('\n\n' + '-'*5 + ' FOLD %i, ' %int(i) + 'HOSPITAL = '+ hospitals_names[i] +'-'*5 + '\n')
        with open(text_file, "a") as file:
            file.write('\n\n' + '-'*5 + ' FOLD %i, ' %int(i) + 'HOSPITAL = '+ hospitals_names[i] +'-'*5 + '\n\n')

        # Use deterministic algorithms if the model type is one of the below
        if cfg.modelType != 'Alexnet' and cfg.modelType != 'VGG':
            torch.use_deterministic_algorithms(True, warn_only = True) 
        
        # Test dataset
        test_idx = i
        test_dataset = copy.deepcopy(fold_datasets[test_idx])
        test_dataset.dataset.transform = transform_chain_basic # Basic transformations and NOT data augmentation
        test_idxs = test_dataset.indices

        # Build another subset excluding test data
        idxs_subfold = np.where(full_dataset.dataset['hospital'] != hospitals_names[i])[0].tolist()
        random.shuffle(idxs_subfold)

        # Train dataset
        train_size = int(0.7 * len(idxs_subfold))  # 70% for train
        train_dataset = Subset(full_dataset, idxs_subfold[:train_size])

        # Validation dataset
        val_size = len(idxs_subfold) - train_size  # 30% for validation
        val_dataset = Subset(full_dataset, idxs_subfold[train_size:])

        # Train dataloader => Suffle
        train_dataloader = DataLoader(train_dataset, batch_size = cfg.train_bs,
                        shuffle = False, num_workers = cfg.train_num_workers)
        
        # Validation dataset => No shuffle
        val_dataloader = DataLoader(val_dataset, batch_size = cfg.val_bs,
                        shuffle = False, num_workers = cfg.val_num_workers)
        
        # Test dataset => No shuffle
        test_dataloader = DataLoader(test_dataset, batch_size = cfg.val_bs,
                        shuffle = False, num_workers = cfg.val_num_workers)
        
        # Create an instance for the CNN model type indicated in the config file
        if hasattr(cfg, 'modelType'):
            if cfg.modelType == 'Resnet18':
                model_ft = Resnet18Model(nMaps,eConfig['rings'], eConfig['arcs'], float(eConfig['max_radius']), float(eConfig['max_angle']), int(eConfig['type']), int(eConfig['delete_last_layer']), int(eConfig['fusion']), pretrained=True, dropout=cfg.dropout)
            elif cfg.modelType == 'Resnet34':
                model_ft = Resnet34Model(nMaps,pretrained=True,dropout=cfg.dropout)
            elif cfg.modelType == 'Resnet50':
                model_ft = Resnet50Model(nMaps,pretrained=True,dropout=cfg.dropout)
            elif cfg.modelType == 'Inception':
                model_ft = InceptionModel(nMaps,pretrained=True,dropout=cfg.dropout)
            elif cfg.modelType == 'RegnetY400m':
                model_ft = RegnetY400mModel(nMaps,pretrained=True,dropout=cfg.dropout)                
            elif cfg.modelType == 'RegnetY800m':
                model_ft = RegnetY800mModel(nMaps,pretrained=True,dropout=cfg.dropout)                
            elif cfg.modelType == 'RegnetY3_2g':
                model_ft = RegnetY3_2gModel(nMaps,pretrained=True,dropout=cfg.dropout)
            elif cfg.modelType == 'RegnetY32g':
                model_ft = RegnetY32gModel(nMaps,pretrained=True,dropout=cfg.dropout)
            elif cfg.modelType == 'Alexnet':
                model_ft = AlexnetModel(nMaps,pretrained=True,dropout=cfg.dropout)
            elif cfg.modelType == 'VGG':
                model_ft = VGGModel(nMaps,pretrained=True,dropout=cfg.dropout)
            elif cfg.modelType == 'Mobilev3large':
                model_ft = Mobilev3large(nMaps,pretrained=True,dropout=cfg.dropout)                            
        else:
            model_ft = Resnet18Model(nMaps,pretrained=True)
        
        # Max number of epochs with no improvement 
        if not hasattr(cfg, 'max_epochs_no_improvement'):
            cfg.max_epochs_no_improvement = 10
        
        model_ft = model_ft.to(device)
        criterion = nn.BCEWithLogitsLoss() # Cross-entropy loss
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
        model_ft = train_model(model_ft, image_datasets, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, text_file,
                               device, int(eConfig['type']), num_epochs=cfg.num_epochs, max_epochs_no_improvement=cfg.max_epochs_no_improvement,min_epochs=10)
    
        # Test model
        test_ids, test_scores,test1_scores, test2_scores, test_labels = test_model(model_ft, test_dataset, test_dataloader, device, int(eConfig['type']))
        
        # Saves the total scores and labels obtained for every test fold 
        # ids and labels by iterations
        total_ids.append(test_ids)
        total_labels.append(test_labels)

        # Logits by iteration
        total_scores.append(test_scores)
        total1_scores.append(test1_scores)
        total2_scores.append(test2_scores)
        
        # AUCs
        AUCs[i] = roc_auc_score(test_labels.flatten(), test_scores.flatten())
        if int(eConfig['type']) == 2:
            AUCs_out1[i] = roc_auc_score(test_labels.flatten(), test1_scores.flatten()) 
            AUCs_out2[i] = roc_auc_score(test_labels.flatten(), test2_scores.flatten())     
        
        # Logits matrix for each id
        scores_matrix[test_idxs, i] = test_scores
        scores1_matrix[test_idxs, i] = test1_scores
        scores2_matrix[test_idxs, i] = test2_scores

        print('testAUC %f \n'%(AUCs[i]))
        print('testAUC (out1) %f \n'%(AUCs_out1[i]))
        print('testAUC (out2) %f \n'%(AUCs_out2[i]))
        with open(text_file, "a") as file:
            # Escribe en el file
            file.write('testAUC %f\n'%(AUCs[i]))
            file.write('testAUC (out1) %f\n'%(AUCs_out1[i]))
            file.write('testAUC (out2) %f\n'%(AUCs_out2[i]))
        
        # Norm scores
        test_scores = test_scores - test_scores.min()
        total_scores_norm.append(test_scores/ test_scores.max()) # Test scores normalizated
        if int(eConfig['type']) == 2:
            test1_scores = test1_scores - test1_scores.min()
            test2_scores = test2_scores - test2_scores.min()
            total1_scores_norm.append(test1_scores/ test1_scores.max()) # Test scores normalizated
            total2_scores_norm.append(test2_scores/ test2_scores.max()) # Test scores normalizated

    hospitals_length = []
    # Show results 
    print('HOSPITAL SEGREGATION FOR TEST DATASET:')
    with open(text_file, "a") as file:
        file.write('\nHOSPITAL SEGREGATION FOR TEST DATASET:\n')
    for i, hospital in enumerate(hospitals_names):
        scores_hospital = total_scores[i]
        hospitals_length.append(len(scores_hospital))
        print(hospital + " n = " + str(hospitals_length[i]) + " AUC = " + str(AUCs[i]) + "\n")
        with open(text_file, "a") as file:
            file.write(hospital + " n = " + str(hospitals_length[i]) + " AUC = " + str(AUCs[i]) + "\n\n")

    # Save results
            
    # Logits for each hospital

    save_results_path = results_path + '/hospital_segregation'
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    # DICT --> Logits for each hospital

    dict_scores = {'hospital': hospitals_names, 'length': hospitals_length,  'img_id': total_ids, 'score': total_scores, 'label': total_labels}
    with open(save_results_path +'/dict_scores.pkl', "wb") as f:
        pickle.dump(dict_scores, f)

        