"""
Created on Tue Apr 18 11:33:40 2023

@author: igonzalez
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
from PIL import Image

# Imports 
# import _init_paths
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader,random_split,ConcatDataset
from torchvision import transforms, utils, models
import pandas as pd
from skimage import io, transform, color, morphology, util
from EyeDataset import EyeDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import cv2
import pdb
import pickle
import time
import copy
# import config as cfg
import random
import numpy.random as npr
import sys
import importlib
import matplotlib.pyplot as plt
    
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

if __name__ == '__main__':

    """ Arguments
        cfg_file: configuration file that contains CNN and training info
        db: database 'HRYC', 'multi' or 'global'
        dir: directory where data directories are saved
        data_dir: directory where data files are saved
        results_dir: directory where results are saved
        seed: random seed
    """
    eConfig = {
        'cfg_file': 'conf.config_70iter',
        'db':'global',
        'dir': 'SEP',
        'data_dir': 'ransac_TH_1.5_r_45',
        'results_dir': 'biomarkers',
        'seed':'0',
        }
    
    args = sys.argv[1::] # Get the name of the config file
    # Assign arguments to variable 'eConfig'
    for i in range(0,len(args),2):
        key = args[i]
        val = args[i+1]
        eConfig[key] = type(eConfig[key])(val)
        print (str(eConfig[key]))
    
    # Reading config file
    cfg = importlib.import_module(eConfig['cfg_file']) # This variable constains all the config file varibales
    
    # CSV and image path
    csvFile='../datasets/' + 'dataset_' + eConfig['db'] + '/' + eConfig['dir'] + '/' + eConfig['data_dir'] + '/annotation_biomarkers.csv'
    imageDir='../datasets/' + 'dataset_' + eConfig['db'] + '/' + eConfig['dir'] + '/' + eConfig['data_dir'] + '/data'

    # Create results directory
    results_dir = 'results/' + eConfig['dir']
    results_path = results_dir + '/' + 'BIOMARKERS' + '/'+ eConfig['results_dir']
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(results_dir + '/' + 'BIOMARKERS') :
        os.makedirs(results_dir + '/'+ 'BIOMARKERS')
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    text_file = results_path + '/'+ eConfig['results_dir'] + '_allData.txt'

    # GPU 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Reading conf from {}'.format(eConfig['cfg_file']))
    print('Using {} device...' .format(device))
    print('Random seed = ' + eConfig['seed'])
    with open(text_file, "w") as file:
            file.write('Reading conf from {}'.format(eConfig['cfg_file']))
            file.write('Using {} device...' .format(device))
            file.write('Random seed = ' + eConfig['seed'])

    iterations = cfg.iterations

    full_dataset=EyeDataset(csvFile, imageDir,transform=None,mapList=None,test=False,random_seed=0)
    
    # Result variables
    total_ids = []
    total_labels = []
    total_scores = []
    scores_matrix = np.full((len(full_dataset),), np.nan)
    total_labels = []
    AUCs = 0

    # Spli dataset in five folds
    full_dataset=EyeDataset(csvFile, imageDir,transform=None,mapList=None,test=False,random_seed=int(eConfig['seed']))

    # Set random seed for reproducibility
    random.seed(int(eConfig['seed']))
    npr.seed(int(eConfig['seed']))
    torch.manual_seed(int(eConfig['seed']))
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True,warn_only = True) # Use deterministic algorithm because its a logistic regressor
    
    # folds loop

    Xtest, Ytest, id_test = full_dataset.getbiomarkers(indices = None)

    # Val dataset
    # val_idx=test_idx+1
    # if val_idx==numFolds:
    #     val_idx=0
    # val_dataset=copy.deepcopy(fold_datasets[val_idx])
    # val_dataset.dataset.transform=transform_chain_basic

    # Concatenate training data
    Xtrain, Ytrain, id_train = full_dataset.getbiomarkers(indices = None) # Aux variables to get the dta
    # train_dataset = ConcatDataset(train_dataset)    
    
    #   Standarization
    scaler = StandardScaler()
    Xtrain_s = scaler.fit_transform(Xtrain)
    Xtest_s = scaler.transform(Xtest)
    
    #   Logistic regressor
    lr = LogisticRegression()
    
    # Cross validation to search best hyperparameters
    param_grid={'C': [1e-3,1e-2,1e-1,1,10,100,1000]}#, 'penalty': ('l2', 'l1','elasticnet')}
    gs_lr = GridSearchCV(lr, param_grid, scoring='roc_auc')
    gs_lr.fit(Xtrain_s, Ytrain)
    print('Best params {}'.format(gs_lr.best_params_)) # Print best params

    # Training regressor
    auc_train = gs_lr.score(Xtrain_s, Ytrain)

    print('\nTrain AUC %f'%auc_train)
    with open(text_file, "a") as file:
        file.write('\n Train AUC %f'%auc_train)
    
    # ids and labels by folds
    total_ids.append(id_test)
    total_labels.append(Ytest)

    # Test regressor
    AUCs= gs_lr.score(Xtest_s, Ytest) # Save test AUC

    # Logits and scores by iteration
    test_scores = gs_lr.predict_proba(Xtest_s)[:,1]
    logits = gs_lr.predict_log_proba(Xtest_s)[:,1]
    total_scores.append(test_scores) # Store total scores
    scores_matrix = logits

    print('\ntestAUC %f'%(AUCs))
    with open(text_file, "a") as file:
        file.write('\ntestAUC %f'%(AUCs))
    
    # Show results
    print('\n\nFinal results:\navgAUC %f\n\n'%(AUCs))
    with open(text_file, "a") as file:
            file.write('\n\n Final results: avgAUC %f\n\n'%(AUCs))
    
    # Save results
    save_results_path = results_path + '/results_nn_allData'
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    
    # Scores norm by iteration
    dict_scores = {'img_id': total_ids, 'score': total_scores, 'label': total_labels}
    with open(save_results_path +'/out_scores.pkl', "wb") as f:
        pickle.dump(dict_scores, f)
    
    # Logits mnatrix
    dict_scores_matrix = {'img_id': full_dataset.dataset['img_id'], 'score': scores_matrix, 'label': full_dataset.dataset['label'] }
    with open(save_results_path +'/dict_scores_matrix.pkl', "wb") as f:
        pickle.dump(dict_scores_matrix, f)