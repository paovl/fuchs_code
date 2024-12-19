
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
from PairEyeDataset import PairEyeDataset
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import numpy as np
import pickle
from torchvision import transforms
from dataAugmentation import ToTensor, reSize
import sys
import importlib
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    eConfig = {
        'dir': 'deactivate70(full DB)',
        'cfg_file':'conf.config1',
        }
    
    #Reading the config file
    cfg = importlib.import_module(eConfig['cfg_file'])
    normalization_file = os.path.join(cfg.db_path,'dataset_global/data/normalization.npz')

    list_transforms=[reSize(cfg.imSize),
                    ToTensor()]

    transform_chain = transforms.Compose(list_transforms)
    full_dataset = PairEyeDataset(cfg.csvFile, cfg.db_path, cfg.imageDir,cfg.dataDir, cfg.error_BFSDir, cfg.error_PACDir, transform = transform_chain, mapList=cfg.mapList,test=False,random_seed=cfg.random_seed,testProp=0)
    patients = full_dataset.dataset['Patient'].unique()
    ojos = full_dataset.dataset['Ojo'].values
    od = sum(ojos == 'OD')
    os = sum(ojos == 'OS')


    print(full_dataset)
    