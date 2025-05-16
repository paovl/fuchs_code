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
        'dir': 'pretrained(original maps)',
        # 'dir' : 'no_deactivate_seed0_dseed0_th0.07_a1000_sigmoidWeightedLoss(full DB)_70train',
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

    data_cnn = np.load( 'results/SEP/ELE_1_ELE_4_CORNEA-DENS_0_PAC_0/medical_rings3_angles3_fusion5_50epochs/'+eConfig['dir']+'/results_nn/dict_raw_scores.pkl', allow_pickle=True)
    # data_lr = np.load('logits_bio.pkl', allow_pickle=True)
    data_lr = np.load('results/SEP/BIOMARKERS/biomarkers/results_nn_5fold/dict_scores_matrix.pkl', allow_pickle=True) # Five fold
    fold = data_lr['fold']
    
    scores_cnn=data_cnn['score'];
    scores_lr=data_lr['score'];
    scores_lr = np.reshape(scores_lr,(-1,1))
    #Normalizando by fold
    for f in range(5):
        media=np.mean(scores_lr[fold==f,:])
        desv=np.std(scores_lr[fold==f,:])
        scores_lr[fold==f,:]=(scores_lr[fold==f,:]-media)/desv

    # Normalization by iteration
    mean_cnn = np.nanmean(scores_cnn,0,keepdims=True);
    std_cnn = np.nanstd(scores_cnn,0,keepdims=True);
    scores_cnn = (scores_cnn - mean_cnn)/std_cnn

    # Normalization by iteration
    mean_lr = np.nanmean(scores_lr,0,keepdims=True);
    std_lr = np.nanstd(scores_lr,0,keepdims=True);
    scores_lr = (scores_lr - mean_lr)/std_lr
    mean_lr = np.nanmean(scores_lr,0,keepdims=True);
    std_lr = np.nanstd(scores_lr,0,keepdims=True);
    scores_lr = (scores_lr - mean_lr)/std_lr

    auc_score_cnn = np.zeros((scores_cnn.shape[1],))
    specificity_cnn = np.zeros((scores_cnn.shape[1],))
    sensitivity_cnn = np.zeros((scores_cnn.shape[1],))
    precision_cnn = np.zeros((scores_cnn.shape[1],))
    youden_cnn = np.zeros((scores_cnn.shape[1],))
    fscore_cnn = np.zeros((scores_cnn.shape[1],))

    labels_cnn=data_cnn['label']
    labels_lr=data_lr['label']
    img_id_cnn=data_cnn['img_id']
    img_id_lr=data_lr['img_id']

    mask = np.isin(img_id_cnn, img_id_lr)

    # Por iteración
    for i in range(scores_cnn.shape[1]):

        outs_cnn_iter = sigmoid(scores_cnn[:,i])

        mask_not_nan_iter = np.isfinite(outs_cnn_iter)

        #Renormalizamos para la fusión
        mean_cnn_iter = np.nanmean(outs_cnn_iter,0,keepdims=True);
        std_cnn_iter = np.nanstd(outs_cnn_iter,0,keepdims=True);
        outs_cnn_iter = (outs_cnn_iter - mean_cnn_iter)/std_cnn_iter

        outs_cnn_iter_bio = outs_cnn_iter[mask & mask_not_nan_iter]
        labels_cnn_iter_bio = labels_cnn[mask & mask_not_nan_iter]
        # outs_cnn_iter_bio = outs_cnn_iter[mask_not_nan_iter]
        # labels_cnn_iter_bio = labels_cnn[mask_not_nan_iter]

        fpr_cnn_iter, tpr_cnn_iter, thresholds_cnn_iter = roc_curve(labels_cnn_iter_bio, outs_cnn_iter_bio)
        # Calcular especificidad (1 - FPR)
        specificity_cnn_iter = 1 - fpr_cnn_iter
        # Calcular AUC (área bajo la curva)
        auc_score_cnn_iter = auc(fpr_cnn_iter, tpr_cnn_iter)
        J_cnn_iter = tpr_cnn_iter + specificity_cnn_iter - 1
        best_idx_cnn_iter =np.argmax(J_cnn_iter)
        precision_cnn_iter = precision_score(labels_cnn_iter_bio, outs_cnn_iter_bio>thresholds_cnn_iter[best_idx_cnn_iter], zero_division=0)

        recall_cnn_iter = tpr_cnn_iter[best_idx_cnn_iter]
        Fscore_cnn_iter = 2*precision_cnn_iter*tpr_cnn_iter/(precision_cnn_iter+tpr_cnn_iter)

        # Añadir resultados
        specificity_cnn[i] = specificity_cnn_iter[best_idx_cnn_iter]
        sensitivity_cnn[i] = recall_cnn_iter
        precision_cnn[i] = precision_cnn_iter
        youden_cnn[i] = J_cnn_iter[best_idx_cnn_iter]
        fscore_cnn[i] = Fscore_cnn_iter[best_idx_cnn_iter]
        auc_score_cnn[i] = auc_score_cnn_iter

        # Print para esta iteracion 
        print('Iter %i AUC %f Sensitivity %f Specificity %f Precision %f J-index %f F-score %f\n' % (i, auc_score_cnn[i], recall_cnn_iter, specificity_cnn_iter[best_idx_cnn_iter], precision_cnn_iter, J_cnn_iter[best_idx_cnn_iter], Fscore_cnn_iter[best_idx_cnn_iter]))

    # Hacer media de todas las metricas
    auc_score_cnn = np.nanmean(auc_score_cnn)
    specificity_cnn = np.nanmean(specificity_cnn)
    sensitivity_cnn = np.nanmean(sensitivity_cnn)
    precision_cnn = np.nanmean(precision_cnn)
    youden_cnn = np.nanmean(youden_cnn)
    fscore_cnn = np.nanmean(fscore_cnn)

    # Print para todas las iteraciones
    print('Mean AUC %f Sensitivity %f Specificity %f Precision %f J-index %f F-score %f\n' % (auc_score_cnn, sensitivity_cnn, specificity_cnn, precision_cnn, youden_cnn, fscore_cnn))