
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
from sklearn.metrics import roc_auc_score, roc_curve, hamming_loss, confusion_matrix
import numpy as np
import cv2
import pdb
import pickle
from dataAugmentation import Normalize, ToTensor, RandomRotation,RandomTranslation,centerMask,centerMinPAC, cropEye, reSize, PolarCoordinates, centerCrop, RandomJitter
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
import pandas as pd
import seaborn as sns

def print_data(data):
    # Title
    print('{:<35}'.format('img_id'), end='')
    for key in data.keys():
        if key != 'img_id':
            print('{:<35}'.format(key), end='')
    print()
    
    # Data
    for i in range(len(data['img_id'])):
        print('{:<35}'.format(data['img_id'][i]), end='')
        for key in data.keys():
            if key != 'img_id':
                print('{:<35.8f}'.format(data[key][i]), end='')
        print()

def write_data(file_path, data):
     with open(file_path, "a") as file:
        # Title
        file.write('{:<35}'.format('img_id'))
        for key in data.keys():
            if key != 'img_id':
                file.write('{:<35}'.format(key))
        file.write('\n')
    
        # Data
        for i in range(len(data['img_id'])):
            file.write('{:<35}'.format(data['img_id'][i]))
            for key in data.keys():
                if key != 'img_id':
                    file.write('{:<35.8f}'.format(data[key][i]))
            file.write('\n')
    

if __name__ == '__main__':
    
    eConfig = {
        'dir': 'deactivate70(full DB)',
        'cfg_file':'conf.config1',
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
    normalization_file = os.path.join(cfg.db_path,'dataset_global/data/normalization.npz')

    if len(cfg.mapList) == 14: 
        patients_val_batch = 'results/' + eConfig['dir'] + '/allMaps' + '/val_batch_patients.pkl'
        text_file = 'results/' + eConfig['dir'] + '/allMaps' + 'val_batch_scores.txt'
        fig_file = 'results/' + eConfig['dir'] + '/allMaps'
    elif len(cfg.mapList) == 1:
        patients_val_batch = 'results/' + eConfig['dir'] + '/' + cfg.mapList[0] + '/val_batch_patients.pkl'
        text_file = 'results/' + eConfig['dir'] + '/' + cfg.mapList[0] + '/val_batch_scores.txt'
        fig_file = 'results/' + eConfig['dir'] + '/' + cfg.mapList[0]
    else:
        patients_val_batch = 'results/' + eConfig['dir'] + '/' + '_'.join(cfg.mapList) + '/val_batch_patients.pkl'
        text_file = 'results/' + eConfig['dir'] + '/' + '_'.join(cfg.mapList) + '/val_batch_scores.txt'
        fig_file = 'results/' + eConfig['dir'] + '/' + '_'.join(cfg.mapList)

    with open(normalization_file, 'rb') as f:
        normalization_data = pickle.load(f)

    with open(patients_val_batch, 'rb') as f:
        patients_batch = pickle.load(f)

    print('Reading scores from dir --> {}'.format(eConfig['dir']))
    with open(text_file, "w") as file:
        file.write('Reading scores from dir --> {}\n\n'.format(eConfig['dir']))

    patients_idx = patients_batch['idx']
    patients_array = np.zeros((len(patients_idx), 2))
    for i in range(len(patients_idx)):
        patients_array[i, 0] = patients_idx[i][0]
        patients_array[i, 1] = patients_idx[i][1]
    
    patients_idx = patients_array
    patients_labels = patients_batch['label']
    patients_outputs = patients_batch['output']

    idxCat = [normalization_data['categories'].index(cat) for cat in cfg.mapList]

    list_transforms=[reSize(cfg.imSize),
                    ToTensor()]

    transform_chain = transforms.Compose(list_transforms)
    full_dataset = PairEyeDataset(cfg.csvFile, cfg.db_path, cfg.imageDir,cfg.dataDir, cfg.error_BFSDir, cfg.error_PACDir, transform = transform_chain, mapList=cfg.mapList,test=False,random_seed=cfg.random_seed,testProp=0)

    patients_eye1 = full_dataset.dataset.iloc[patients_idx[:,0].astype(int)]['Ojo'].values
    patients_eye2 = full_dataset.dataset.iloc[patients_idx[:,1].astype(int)]['Ojo'].values

    patients_session1 = full_dataset.dataset.iloc[patients_idx[:,0].astype(int)]['Sesion'].values
    patients_session2 = full_dataset.dataset.iloc[patients_idx[:,1].astype(int)]['Sesion'].values

    patients_id1 = full_dataset.dataset.iloc[patients_idx[:,0].astype(int)]['Patient'].values
    patients_id2 = full_dataset.dataset.iloc[patients_idx[:,1].astype(int)]['Patient'].values
    # Calcular la diferencia absoluta entre los scores y las etiquetas
    diff = np.abs(patients_outputs- patients_labels)

    # Crear un DataFrame para almacenar los datos con el ranking
    img_id = []
    for i in range(patients_idx.shape[0]):
        img_id.append(str(patients_id1[i]) + '_' + patients_eye1[i] + '_' + patients_session1[i] + '-' + str(patients_id2[i]) + '_' + patients_eye2[i] + '_' + patients_session2[i])
    df = pd.DataFrame({'img_id': img_id, 'Score': patients_outputs, 'True Label': patients_labels, 'Difference': diff})

    # Ordenar el DataFrame por la columna 'Difference' (de menor a mayor)
    df_sorted = df.sort_values(by='Difference', ascending=True).reset_index(drop=True)

    print_data(df_sorted)
    write_data(text_file, df_sorted)

    # Calcular la curva ROC
    fpr, tpr, thresholds = roc_curve(patients_labels, patients_outputs)

    # Calcular el área bajo la curva (AUC)
    roc_auc = roc_auc_score(patients_labels, patients_outputs)

    # Graficar la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)  # Línea diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(fig_file + '/roc_auc.png', dpi=300, bbox_inches='tight')

    # Confusion matrix
    patients_labels_outputs = (patients_outputs >= 0.5).astype(int)

    cm = confusion_matrix(patients_labels.astype(int), patients_labels_outputs)

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Mostrar la matriz de confusión
    cax = ax.matshow(cm, cmap='viridis')

    # Añadir etiquetas
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")

    # Agregar valores de la matriz en las celdas
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center', color='white')

    # Configurar los ticks
    plt.xticks([0, 1], ['Pred. Negativo', 'Pred. Positivo'])
    plt.yticks([0, 1], ['Real Negativo', 'Real Positivo'])

    plt.colorbar(cax)
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.title("Matriz de Confusión")
    plt.savefig(fig_file + '/confusion_matrix.png', dpi=300, bbox_inches='tight')

    


