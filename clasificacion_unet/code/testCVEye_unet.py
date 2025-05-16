
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
from dataAugmentation_unet import Normalize, ToTensor, RandomRotation,RandomTranslation,centerMask,centerMinPAC, cropEye, reSize, PolarCoordinates, centerCrop, RandomJitter
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
        'cfg_file' : 'conf.config_70iter_PAC_0',
        'db': 'global',
        'bio':'0',
        'dir': 'SEP',
        'data_dir': 'ransac_TH_1.5_r_45',
        'results_dir': 'medical_rings3_angles3_fusion5_50epochs',
        'seed' : '0',
        'dir_weights_file':'none',
        'weights_seed': '0',
        'weights_dseed': '0',
        'max_angle': '360',
        'weights_ssl_file': 'unet_self_supervised_fold.pth', 
        'dupe': '1'
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
        if int(eConfig['dupe']) == 1:
            csvFile='../datasets/' + 'dataset_' + eConfig['db'] + '/' + eConfig['dir'] + '/' + eConfig['data_dir'] + '/annotation_dupe.csv'
        else:
            csvFile='../datasets/' + 'dataset_' + eConfig['db'] + '/' + eConfig['dir'] + '/' + eConfig['data_dir'] + '/annotation.csv'
    if int(eConfig['dupe']) == 1:
        imageDir = db_path + '/'+ dir_file + '/' + eConfig['data_dir'] + '/data_dupe'
    else:
        imageDir = db_path + '/'+ dir_file + '/' + eConfig['data_dir'] + '/data'

    # Create results directory
    results_dir = 'results/' +  eConfig['dir']
    if len(cfg.mapList) == 14: 
        results_path = results_dir + '/' + 'allMaps' + '/'+ eConfig['results_dir']
    elif len(cfg.mapList) == 1:
        results_path = results_dir + '/' + cfg.mapList[0] + '/'+ eConfig['results_dir']
    else:
        results_path = results_dir + '/' + '_'.join(cfg.mapList) + '/'+ eConfig['results_dir']
    
    #only to test SSL

    results_path = results_path + '/' + eConfig['dir_weights_file']

    text_file = results_path + '/test_after.txt'

    model_weights_path = results_path + '/model_weights'

    # Read ssl model
    path_weights_file = '../../ssl_unet/code/results/'

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

    # Training iterations
    iterations = cfg.iterations

    # Result variables 
    total_scores=[]
    total_ids =[]
    total_labels = []
    AUCs = np.zeros((iterations,))

    full_dataset = EyeDataset(csvFile, imageDir, transform = transform_chain_basic, mapList = cfg.mapList, test = False, random_seed = int(eConfig['seed']), testProp = 0)
    
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
        full_dataset = EyeDataset(csvFile, imageDir, transform = transform_chain_basic, mapList = cfg.mapList, test = False, random_seed = random_seed, testProp = 0)
        random_generator = torch.Generator().manual_seed(random_seed)
        dataset_sizes = len(full_dataset)*np.ones(cfg.numFolds)//float(cfg.numFolds) # Fold size
        remSamples = len(full_dataset)-int(dataset_sizes.sum()) # Remaining samples
        for i in range(remSamples):
            dataset_sizes[i] += 1
        fold_datasets = random_split(full_dataset, dataset_sizes.astype(int), generator = random_generator)

        print('\n\n' + '-'*5 + ' ITER %i ' %int(iter) + '-'*5 + '\n')
        with open(text_file, "a") as file:
            file.write('\n\n' + '-'*5 + ' ITER %i ' %int(iter) + '-'*5 + '\n')

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
        
        # Test dataloader => No shuffle
        test_dataloader = DataLoader(test_dataset, batch_size = cfg.val_bs,
                        shuffle = False, num_workers = cfg.val_num_workers)
        
        # Create an instance for the CNN model type
        model_ft = UNetModel(nMaps, 1, model_ssl)

        # Load weights
        model_weights_path_iter = model_weights_path + '/model_weights_iter' + str(iter) + '.pth'
        model_ft.load_state_dict(torch.load(model_weights_path_iter))
        model_ft.eval() 

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
            file.write('testAUC %f \n'%(AUCs[iter]))
 
        # Norm scores
        test_scores = test_scores - test_scores.min()
        
    # Show results
    print('mean testAUC accross iterations: %f \n'%(np.mean(AUCs)))
    with open(text_file, "a") as file:
        file.write('mean testAUC accross iterations: %f \n'%(np.mean(AUCs)))

    save_results_path = results_path + '/results_nn'

    # DICT --> Logits matrix

    dict_raw_scores = {'img_id': all_ids, 'score': scores_matrix, 'label': all_labels}
    with open(save_results_path +'/dict_raw_scores_test.pkl', "wb") as f:
        pickle.dump(dict_raw_scores, f)
    