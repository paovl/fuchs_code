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

def sigmoid(x):
    return 1.0/(1+np.exp(-x));

if __name__ == '__main__': 

    eConfig = {
        'dir' : 'no_deactivate_seed0_dseed0_th0.07_a1000_sigmoidWeightedLoss(full DB)',
    }

    args = sys.argv[1::]
    for i in range(0,len(args),2):
        key = args[i]
        val = args[i+1]
        eConfig[key] = type(eConfig[key])(val)
        print (str(eConfig[key]))

    # data_cnn = np.load('logits_cnn_dbBio.pkl', allow_pickle=True)
    # data_cnn = np.load('logits_cnn.pkl', allow_pickle=True)

    print('Loading data from: ', eConfig['dir'])

    data_cnn = np.load( 'results/SEP/ELE_4_CORNEA-DENS_0_PAC_0/medical_rings3_angles3_fusion5_50epochs/'+eConfig['dir']+'/results_nn/dict_raw_scores.pkl', allow_pickle=True)
    # data_lr = np.load('logits_bio.pkl', allow_pickle=True)
    data_lr = np.load('results/SEP/BIOMARKERS/biomarkers/results_nn_5fold/dict_scores_matrix.pkl', allow_pickle=True)
    fold = data_lr['fold']
    
    scores_cnn=data_cnn['score'];
    scores_lr=data_lr['score'];
    scores_lr = np.reshape(scores_lr,(-1,1))
    #Normalizando by fold
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
    mean_cnn = np.mean(outs_cnn,0,keepdims=True);
    std_cnn = np.std(outs_cnn,0,keepdims=True);
    outs_cnn = (outs_cnn - mean_cnn)/std_cnn
    mean_lr = np.mean(outs_lr,0,keepdims=True);
    std_lr = np.std(outs_lr,0,keepdims=True);
    outs_lr = (outs_lr - mean_lr)/std_lr
    
    labels_cnn=data_cnn['label']
    labels_lr=data_lr['label']
    img_id_cnn=data_cnn['img_id']
    img_id_lr=data_lr['img_id']
    
    # mask = np.isin(img_id_cnn, img_id_lr)
    # outs_cnn = outs_cnn[mask]
    # labels_cnn = labels_cnn[mask]
    
    #Fusión simple
    alpha=0.5
    # outs_fus = alpha*outs_cnn + (1-alpha)*outs_lr
    outs_fus =  outs_lr
    
    # Obtener FPR (false positive rate), TPR (true positive rate) y umbrales
    fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(labels_cnn, outs_cnn)
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
    # best_idx_lr = np.argmin(abs(specificity_lr-specificity_cnn[best_idx_cnn]))
    # best_idx_fus = np.argmin(abs(specificity_fus-specificity_cnn[best_idx_cnn]))
    
    precision_cnn = precision_score(labels_cnn, outs_cnn>thresholds_cnn[best_idx_cnn], zero_division=0)
    precision_lr = precision_score(labels_lr, outs_lr>thresholds_lr[best_idx_lr], zero_division=0)
    precision_fus = precision_score(labels_lr, outs_fus>thresholds_fus[best_idx_fus], zero_division=0)
    
    recall_cnn = tpr_cnn[best_idx_cnn]
    recall_lr = tpr_lr[best_idx_lr]
    recall_fus = tpr_fus[best_idx_fus]
    
    Fscore_cnn = 2*precision_cnn*tpr_cnn/(precision_cnn+tpr_cnn)
    Fscore_lr = 2*precision_lr*tpr_lr/(precision_lr+tpr_lr)
    Fscore_fus = 2*precision_fus*tpr_fus/(precision_fus+tpr_fus)
    
    print('Best CNN AUC {} Sensitivity {} Specificity {} Precision {} J-index {} F-score {}'.format(auc_score_cnn,recall_cnn,specificity_cnn[best_idx_cnn],precision_cnn,J_cnn[best_idx_cnn],Fscore_cnn[best_idx_cnn]))
    print('Best Lr AUC {} Sensitivity {} Specificity {} Precision {} J-index {} F-score {}'.format(auc_score_lr,recall_lr,specificity_lr[best_idx_lr],precision_lr,J_lr[best_idx_lr],Fscore_lr[best_idx_lr]))
    print('Best Fusion AUC {} Sensitivity {} Specificity {} Precision {} J-index {} F-score {}'.format(auc_score_fus,recall_fus,specificity_fus[best_idx_fus],precision_fus,J_fus[best_idx_fus],Fscore_fus[best_idx_fus]))
    
    plt.figure(figsize=(8, 6))
    plt.plot(specificity_cnn, tpr_cnn, label=f'AUC (ours) = {auc_score_cnn:.2f}', color='blue')
    plt.plot(specificity_lr, tpr_lr, label=f'AUC (Arnalich2019)= {auc_score_lr:.2f}', color='red')
    plt.plot(specificity_fus, tpr_fus, label=f'AUC (Fusion)= {auc_score_fus:.2f}', color='orange')
    plt.xlabel('Especificidad (1 - FPR)')
    plt.ylabel('Sensibilidad (TPR)')
    plt.title('Curva Sensibilidad vs Especificidad')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.clf()
