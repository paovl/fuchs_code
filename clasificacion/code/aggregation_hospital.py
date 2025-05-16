#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 10:09:35 2025

@author: ivan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,precision_score
import pdb
import sys
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Rekha"

def print_data(data):
    # Title
    print('{:<12}'.format('img_id'), end='')
    for key in data.keys():
        if key != 'img_id':
            print('{:<12}'.format(key), end='')
    print()
    
    # Data
    for i in range(len(data['img_id'])):
        print('{:<12}'.format(data['img_id'][i]), end='')
        for key in data.keys():
            if key != 'img_id':
                print('{:<12.8f}'.format(data[key][i]), end='')
        print()

def sigmoid(x):
    return 1.0/(1+np.exp(-x));

if __name__ == '__main__':

    cnn_file = 'logits_th0.07_a1000_gm.pkl'
    lr_file = 'logits_bio_5fold.pkl'

    data_cnn = np.load(cnn_file, allow_pickle=True)
    data_lr = np.load(lr_file, allow_pickle=True)
    fold = data_lr['fold']

    print("Loading data from: ", cnn_file )

    scores_cnn=data_cnn['score']
    scores_lr=data_lr['score']

    scores_lr = np.reshape(scores_lr,(-1,1))
    
    #Normalizando scores del lr by fold
    for f in range(5):
        media=np.mean(scores_lr[fold==f,:])
        desv=np.std(scores_lr[fold==f,:])
        scores_lr[fold==f,:]=(scores_lr[fold==f,:]-media)/desv

    mean_cnn = np.nanmean(scores_cnn,0,keepdims=True);
    std_cnn = np.nanstd(scores_cnn,0,keepdims=True);
    scores_cnn = (scores_cnn - mean_cnn)/std_cnn
    mean_lr = np.nanmean(scores_lr,0,keepdims=True);
    std_lr = np.nanstd(scores_lr,0,keepdims=True);
    scores_lr = (scores_lr - mean_lr)/std_lr

    outs_cnn = np.nanmean(sigmoid(scores_cnn),1);
    outs_lr = np.nanmean(sigmoid(scores_lr),1);

    #Renormalizamos para la fusión
    mean_cnn = np.nanmean(outs_cnn,0,keepdims=True);
    std_cnn = np.nanstd(outs_cnn,0,keepdims=True);
    outs_cnn = (outs_cnn - mean_cnn)/std_cnn
    mean_lr = np.mean(outs_lr,0,keepdims=True);
    std_lr = np.std(outs_lr,0,keepdims=True);
    outs_lr = (outs_lr - mean_lr)/std_lr

    labels_cnn=data_cnn['label']
    labels_lr=data_lr['label']
    img_id_cnn=data_cnn['img_id']
    img_id_lr=data_lr['img_id']
    
    mask = np.isin(img_id_cnn, img_id_lr)

    # outs_cnn = outs_cnn[mask]
    # labels_cnn = labels_cnn[mask]

    #Fusión simple
    alpha=0.5
    # outs_fus = alpha*outs_cnn + (1-alpha)*outs_lr
    outs_fus = outs_lr

    # Obtener FPR (false positive rate), TPR (true positive rate) y umbrales
    mask_not_nan = np.isfinite(outs_cnn) # Esto para en el caso de los hospitales que solo se tienen datos de test para determinados pacientes
    print('Number of trues in mask:', np.sum(mask_not_nan)) # imprimir numero de trues en mask
    fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(labels_cnn[mask_not_nan], outs_cnn[mask_not_nan])
    # Calcular especificidad (1 - FPR)
    specificity_cnn = 1 - fpr_cnn
    # Calcular AUC (área bajo la curva)
    auc_score_cnn = auc(fpr_cnn, tpr_cnn)
    
    # Obtener FPR (false positive rate), TPR (true positive rate) y umbrales
    fpr_lr, tpr_lr, thresholds_lr= roc_curve(labels_lr, outs_lr)
    # Calcular especificidad (1 - FPR)
    specificity_lr = 1 - fpr_lr
    # Calcular AUC (área bajo la curva)
    auc_score_lr = auc(fpr_lr, tpr_lr)
    
    # Obtener FPR (false positive rate), TPR (true positive rate) y umbrales
    fpr_fus, tpr_fus, thresholds_fus= roc_curve(labels_lr, outs_fus)
    # Calcular especificidad (1 - FPR)
    specificity_fus = 1 - fpr_fus
    # Calcular AUC (área bajo la curva)
    auc_score_fus = auc(fpr_fus, tpr_fus)

    #Índice J (índice de Youden)
    J_cnn = tpr_cnn + specificity_cnn - 1
    J_lr = tpr_lr + specificity_lr - 1
    J_fus = tpr_fus + specificity_fus - 1

    best_idx_cnn=np.argmax(J_cnn)
    best_idx_lr=np.argmax(J_lr)
    best_idx_fus=np.argmax(J_fus)

    precision_cnn = precision_score(labels_cnn, outs_cnn>thresholds_cnn[best_idx_cnn], zero_division=0)
    precision_lr = precision_score(labels_lr, outs_lr>thresholds_lr[best_idx_lr], zero_division=0)
    precision_fus = precision_score(labels_lr, outs_fus>thresholds_fus[best_idx_fus], zero_division=0)

    recall_cnn = tpr_cnn[best_idx_cnn]
    recall_lr = tpr_lr[best_idx_lr]
    recall_fus = tpr_fus[best_idx_fus]
    
    Fscore_cnn = 2*precision_cnn*tpr_cnn/(precision_cnn+tpr_cnn)
    Fscore_lr = 2*precision_lr*tpr_lr/(precision_lr+tpr_lr)
    Fscore_fus = 2*precision_fus*tpr_fus/(precision_fus+tpr_fus)

    print('Threshold  CNN: ', sigmoid(thresholds_cnn[best_idx_cnn]))
    print('Threshold  LR: ', sigmoid(thresholds_lr[best_idx_lr]))
    
    print('Best CNN AUC {} Sensitivity {} Specificity {} Precision {} J-index {} F-score {}'.format(auc_score_cnn,recall_cnn,specificity_cnn[best_idx_cnn],precision_cnn,J_cnn[best_idx_cnn],Fscore_cnn[best_idx_cnn]))
    print('Best Lr AUC {} Sensitivity {} Specificity {} Precision {} J-index {} F-score {}'.format(auc_score_lr,recall_lr,specificity_lr[best_idx_lr],precision_lr,J_lr[best_idx_lr],Fscore_lr[best_idx_lr]))
    print('Best Fusion AUC {} Sensitivity {} Specificity {} Precision {} J-index {} F-score {}'.format(auc_score_fus,recall_fus,specificity_fus[best_idx_fus],precision_fus,J_fus[best_idx_fus],Fscore_fus[best_idx_fus]))
