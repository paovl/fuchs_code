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

    eConfig = {
        # 'dir': 'pretrained',
        'dir' : 'no_deactivate_seed0_dseed0_th0.07_a1000_sigmoidWeightedLoss(full DB)_70train',
        'dir_baseline': 'pretrained(original maps)_5fold'
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
    print('Loading data from: ', eConfig['dir_baseline'])

    data_cnn_baseline = np.load( 'results/SEP/ELE_1_ELE_4_CORNEA-DENS_0_PAC_0/medical_rings3_angles3_fusion5_50epochs/'+eConfig['dir_baseline']+'/results_nn_5folds/dict_raw_scores.pkl', allow_pickle=True)
    data_cnn = np.load( 'results/SEP/ELE_4_CORNEA-DENS_0_PAC_0/medical_rings3_angles3_fusion5_50epochs/'+eConfig['dir']+'/results_nn/dict_raw_scores.pkl', allow_pickle=True)
    data_lr = np.load('results/SEP/BIOMARKERS/biomarkers/results_nn_5fold/dict_scores_matrix.pkl', allow_pickle=True)
    fold = data_lr['fold']
    fold_baseline = data_cnn_baseline['fold']

    # pdb.set_trace()

    scores_cnn_baseline=data_cnn_baseline['score']
    scores_cnn=data_cnn['score']
    scores_lr=data_lr['score']

    scores_lr = np.reshape(scores_lr,(-1,1))
    scores_cnn_baseline = np.reshape(scores_cnn_baseline,(-1,1))

    #Normalizando scores del lr by fold
    for f in range(5):
        media=np.mean(scores_lr[fold==f,:])
        desv=np.std(scores_lr[fold==f,:])
        scores_lr[fold==f,:]=(scores_lr[fold==f,:]-media)/desv

        media_baseline=np.mean(scores_cnn_baseline[fold_baseline==f,:])
        desv_baseline=np.std(scores_cnn_baseline[fold_baseline==f,:])
        scores_cnn_baseline[fold_baseline==f,:]=(scores_cnn_baseline[fold_baseline==f,:]-media_baseline)/desv_baseline

    mean_cnn_baseline = np.nanmean(scores_cnn_baseline,0,keepdims=True);
    std_cnn_baseline = np.nanstd(scores_cnn_baseline,0,keepdims=True);
    scores_cnn_baseline = (scores_cnn_baseline - mean_cnn_baseline)/std_cnn_baseline
    mean_cnn = np.nanmean(scores_cnn,0,keepdims=True);
    std_cnn = np.nanstd(scores_cnn,0,keepdims=True);
    scores_cnn = (scores_cnn - mean_cnn)/std_cnn
    mean_lr = np.nanmean(scores_lr,0,keepdims=True);
    std_lr = np.nanstd(scores_lr,0,keepdims=True);
    scores_lr = (scores_lr - mean_lr)/std_lr

    outs_cnn_baseline = np.nanmean(sigmoid(scores_cnn_baseline),1);
    outs_cnn = np.nanmean(sigmoid(scores_cnn),1);
    outs_lr = np.nanmean(sigmoid(scores_lr),1);

    #Renormalizamos para la fusión
    mean_cnn_baseline = np.mean(outs_cnn_baseline,0,keepdims=True)
    std_cnn_baseline = np.std(outs_cnn_baseline,0,keepdims=True)
    outs_cnn_baseline = (outs_cnn_baseline - mean_cnn_baseline)/std_cnn_baseline
    mean_cnn = np.nanmean(outs_cnn,0,keepdims=True);
    std_cnn = np.nanstd(outs_cnn,0,keepdims=True);
    outs_cnn = (outs_cnn - mean_cnn)/std_cnn
    mean_lr = np.mean(outs_lr,0,keepdims=True);
    std_lr = np.std(outs_lr,0,keepdims=True);
    outs_lr = (outs_lr - mean_lr)/std_lr

    labels_cnn_baseline=data_cnn_baseline['label']
    img_id_cnn_baseline=data_cnn_baseline['img_id']
    labels_cnn=data_cnn['label']
    labels_lr=data_lr['label']
    img_id_cnn=data_cnn['img_id']
    img_id_lr=data_lr['img_id']
    
    mask = np.isin(img_id_cnn, img_id_lr)

    #Fusión simple
    alpha=0.5
    # outs_fus = alpha*outs_cnn + (1-alpha)*outs_lr
    outs_fus = outs_lr

    # Baseline 
    fpr_cnn_baseline, tpr_cnn_baseline, thresholds_cnn_baseline = roc_curve(labels_cnn_baseline, outs_cnn_baseline)
    specificity_cnn_baseline = 1 - fpr_cnn_baseline
    auc_score_cnn_baseline = auc(fpr_cnn_baseline, tpr_cnn_baseline)
    
    # Obtener FPR (false positive rate), TPR (true positive rate) y umbrales
    mask_not_nan = np.isfinite(outs_cnn)
    # imprimir numero de trues en mask
    print('Number of trues in mask:', np.sum(mask_not_nan))
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
    J_cnn_baseline = tpr_cnn_baseline + specificity_cnn_baseline - 1
    J_cnn = tpr_cnn + specificity_cnn - 1
    J_lr = tpr_lr + specificity_lr - 1
    J_fus = tpr_fus + specificity_fus - 1

    best_idx_cnn_baseline=np.argmax(J_cnn_baseline)
    best_idx_cnn=np.argmax(J_cnn)
    best_idx_lr=np.argmax(J_lr)
    best_idx_fus=np.argmax(J_fus)

    idx_cnn_specificity_97 = np.argmin(np.abs(specificity_cnn - 0.97))
    idx_cnn_sensitivity_97 = np.argmin(np.abs(tpr_cnn - 0.97))

    precision_cnn_baseline = precision_score(labels_cnn_baseline, outs_cnn_baseline>thresholds_cnn_baseline[best_idx_cnn_baseline], zero_division=0)
    precision_cnn = precision_score(labels_cnn, outs_cnn>thresholds_cnn[best_idx_cnn], zero_division=0)
    precision_cnn_specificity_97 = precision_score(labels_cnn, outs_cnn>thresholds_cnn[idx_cnn_specificity_97], zero_division=0)
    precision_cnn_sensitivity_97 = precision_score(labels_cnn, outs_cnn>thresholds_cnn[idx_cnn_sensitivity_97], zero_division=0)
    precision_lr = precision_score(labels_lr, outs_lr>thresholds_lr[best_idx_lr], zero_division=0)
    precision_fus = precision_score(labels_lr, outs_fus>thresholds_fus[best_idx_fus], zero_division=0)

    recall_cnn_baseline = tpr_cnn_baseline[best_idx_cnn_baseline]
    recall_cnn = tpr_cnn[best_idx_cnn]
    recall_cnn_specificity_97 = tpr_cnn[idx_cnn_specificity_97]
    recall_cnn_sensitivity_97 = tpr_cnn[idx_cnn_sensitivity_97]
    recall_lr = tpr_lr[best_idx_lr]
    recall_fus = tpr_fus[best_idx_fus]
    
    Fscore_cnn_baseline = 2*precision_cnn_baseline*tpr_cnn_baseline/(precision_cnn_baseline+tpr_cnn_baseline)
    Fscore_cnn = 2*precision_cnn*tpr_cnn/(precision_cnn+tpr_cnn)
    Fscore_cnn_specificity_97 = 2*precision_cnn_specificity_97*tpr_cnn/(precision_cnn_specificity_97+tpr_cnn)
    Fscore_cnn_sensitivity_97 = 2*precision_cnn_sensitivity_97*tpr_cnn/(precision_cnn_sensitivity_97+tpr_cnn)
    Fscore_lr = 2*precision_lr*tpr_lr/(precision_lr+tpr_lr)
    Fscore_fus = 2*precision_fus*tpr_fus/(precision_fus+tpr_fus)

    print('Threshold CNN Baseline: ', sigmoid(thresholds_cnn_baseline[best_idx_cnn_baseline]))
    print('Threshold  CNN: ', sigmoid(thresholds_cnn[best_idx_cnn]))
    print('Threshold  LR: ', sigmoid(thresholds_lr[best_idx_lr]))
    
    print('Best CNN Baseline AUC {} Sensitivity {} Specificity {} Precision {} J-index {} F-score {}'.format(auc_score_cnn_baseline,recall_cnn_baseline,specificity_cnn_baseline[best_idx_cnn_baseline],precision_cnn_baseline,J_cnn_baseline[best_idx_cnn_baseline],Fscore_cnn_baseline[best_idx_cnn_baseline]))
    print('Best CNN AUC {} Sensitivity {} Specificity {} Precision {} J-index {} F-score {}'.format(auc_score_cnn,recall_cnn,specificity_cnn[best_idx_cnn],precision_cnn,J_cnn[best_idx_cnn],Fscore_cnn[best_idx_cnn]))
    print('Best Lr AUC {} Sensitivity {} Specificity {} Precision {} J-index {} F-score {}'.format(auc_score_lr,recall_lr,specificity_lr[best_idx_lr],precision_lr,J_lr[best_idx_lr],Fscore_lr[best_idx_lr]))
    print('Best Fusion AUC {} Sensitivity {} Specificity {} Precision {} J-index {} F-score {}'.format(auc_score_fus,recall_fus,specificity_fus[best_idx_fus],precision_fus,J_fus[best_idx_fus],Fscore_fus[best_idx_fus]))
    # Specificity 97
    print('Threshold CNN Specificity 97: ', sigmoid(thresholds_cnn[idx_cnn_specificity_97]))
    print('CNN Specificity 97 Specificity {} Sensitivity {} Precision {} F-score {}'.format(specificity_cnn[idx_cnn_specificity_97],recall_cnn_specificity_97,precision_cnn_specificity_97,Fscore_cnn_specificity_97[idx_cnn_specificity_97]))

    # Sensitivity 97
    print('Threshold CNN Sensitivity 97: ', sigmoid(thresholds_cnn[idx_cnn_sensitivity_97]))
    print('CNN Sensitivity 97 Specificity {} Sensitivity {} Precision {} F-score {}'.format(specificity_cnn[idx_cnn_sensitivity_97],recall_cnn_sensitivity_97,precision_cnn_sensitivity_97,Fscore_cnn_sensitivity_97[idx_cnn_sensitivity_97]))

    # Colocando umbral en 0.5 imprimri falsos positivos, FALSOS NEGATIVOS, TRUE POSITIVOS, TRUE NEGATIVES
    #print ids and scores and labels --> false negatives with thereshold 0.5
    # scores_cnn = sigmoid(outs_cnn)
    # predict_labels_cnn = scores_cnn > 0.5
    # false_negatives = {'img_id': [], 'score': [], 'label': []}
    # for i in range(len(img_id_cnn)):
    #     if labels_cnn[i] == 1 and predict_labels_cnn[i] == 0:
    #         false_negatives['img_id'].append(img_id_cnn[i])
    #         false_negatives['score'].append(scores_cnn[i])
    #         false_negatives['label'].append(labels_cnn[i])
    # print("\nFALSE NEGATIVES (THRESHOLD 0.5)")
    # print_data(false_negatives)

    # #print img_id_cnn and scores and labels_cnn --> false positives with thereshold 0.5
    # false_positives = {'img_id': [], 'score': [], 'label': []}
    # for i in range(len(img_id_cnn)):
    #     if labels_cnn[i] == 0 and predict_labels_cnn[i] == 1:
    #         false_positives['img_id'].append(img_id_cnn[i])
    #         false_positives['score'].append(scores_cnn[i])
    #         false_positives['label'].append(labels_cnn[i])
    # print("\nFALSE POSITIVES (THRESHOLD 0.5)")
    # print_data(false_positives)

    # # print img_id_cnn and scores and labels_cnn --> true positives with thereshold 0.5
    # true_positives = {'img_id': [], 'score': [], 'label': []}
    # for i in range(len(img_id_cnn)):
    #     if labels_cnn[i] == 1 and predict_labels_cnn[i] == 1:
    #         true_positives['img_id'].append(img_id_cnn[i])
    #         true_positives['score'].append(scores_cnn[i])
    #         true_positives['label'].append(labels_cnn[i])
    # print("\nTRUE POSITIVES (THRESHOLD 0.5)")
    # print_data(true_positives)

    # # print img_id_cnn and scores and labels_cnn --> true negatives with thereshold 0.5
    # true_negatives = {'img_id': [], 'score': [], 'label': []}
    # for i in range(len(img_id_cnn)):
    #     if labels_cnn[i] == 0 and predict_labels_cnn[i] == 0:
    #         true_negatives['img_id'].append(img_id_cnn[i])
    #         true_negatives['score'].append(scores_cnn[i])
    #         true_negatives['label'].append(labels_cnn[i])
    # print("\nTRUE NEGATIVES (THRESHOLD 0.5)")
    # print_data(true_negatives)

    plt.figure(figsize=(8, 6))

    # Curvas principales
    plt.plot(specificity_cnn_baseline, tpr_cnn_baseline, label=f'AUC (baseline) = {auc_score_cnn_baseline:.2f}', color='green')
    plt.plot(specificity_cnn, tpr_cnn, label=f'AUC (ours) = {auc_score_cnn:.2f}', color='orange')
    plt.plot(specificity_lr, tpr_lr, label=f'AUC (Arnalich2019)= {auc_score_lr:.2f}', color='red')

    # Puntos "Best" sin etiqueta
    plt.scatter(specificity_cnn_baseline[best_idx_cnn_baseline], tpr_cnn_baseline[best_idx_cnn_baseline], color='black', s = 70, marker = 'x', alpha=1)
    plt.scatter(specificity_cnn[best_idx_cnn], tpr_cnn[best_idx_cnn], color='black',  s = 70, marker = 'x', alpha=1)
    plt.scatter(specificity_lr[best_idx_lr], tpr_lr[best_idx_lr], color='black',  s = 70, marker = 'x', alpha=1)

    # Agregar un solo ítem a la leyenda para representar todos los puntos naranjas
    custom_legend = [
        Line2D([0], [0], color='green', label=f'AUC (baseline) = {auc_score_cnn_baseline:.2f}'),
        Line2D([0], [0], color='red', label=f'AUC (Arnalich2019) = {auc_score_lr:.2f}'),
        Line2D([0], [0], color='orange', label=f'AUC (ours) = {auc_score_cnn:.2f}'),
        Line2D([0], [0], marker='x', color='black', markerfacecolor='black', markersize=10, label='Max J-index')  # Punto ficticio para leyenda
    ]

    plt.xlabel('Specificity (1 - FPR)', fontsize=14)
    plt.ylabel('Sensitivity (TPR)', fontsize=14)
    plt.title('Sensitivity vs Specificity', fontsize=16)
    plt.grid(True)
    plt.legend(handles=custom_legend, fontsize=12)
    plt.tight_layout()
    plt.savefig("sensitivity_vs_specificity.png", dpi=300)
    plt.savefig("sensitivity_vs_specificity.pdf", dpi=300)
    plt.clf()

    plt.figure(figsize=(8, 6))

    # Curvas principales
    plt.plot(fpr_cnn_baseline, tpr_cnn_baseline, label=f'AUC (baseline) = {auc_score_cnn_baseline:.2f}', color='green')
    plt.plot(fpr_cnn, tpr_cnn, label=f'AUC (ours) = {auc_score_cnn:.2f}', color='orange')
    plt.plot(fpr_lr, tpr_lr, label=f'AUC (Arnalich2019)= {auc_score_lr:.2f}', color='red')

    # Puntos "Best" sin etiqueta
    plt.scatter(fpr_cnn_baseline[best_idx_cnn_baseline], tpr_cnn_baseline[best_idx_cnn_baseline], color='black', s = 70, marker = 'x', alpha=1)
    plt.scatter(fpr_cnn[best_idx_cnn], tpr_cnn[best_idx_cnn], color='black',  s = 70, marker = 'x', alpha=1)
    plt.scatter(fpr_lr[best_idx_lr], tpr_lr[best_idx_lr], color='black',  s = 70, marker = 'x', alpha=1)

    # Agregar un solo ítem a la leyenda para representar todos los puntos naranjas
    custom_legend = [
        Line2D([0], [0], color='green', label=f'AUC (baseline) = {auc_score_cnn_baseline:.2f}'),
        Line2D([0], [0], color='red', label=f'AUC (Arnalich2019) = {auc_score_lr:.2f}'),
        Line2D([0], [0], color='orange', label=f'AUC (ours) = {auc_score_cnn:.2f}'),
        Line2D([0], [0], marker='x', color='black', markerfacecolor='black', markersize=10, label='Max J-index')  # Punto ficticio para leyenda
    ]

    plt.xlabel('1 - Specificity (FPR)', fontsize=14)
    plt.ylabel('Sensitivity (TPR)', fontsize=14)
    plt.title('ROC curve', fontsize=16)
    plt.grid(True)
    plt.legend(handles=custom_legend, fontsize=12)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300)
    plt.savefig("roc_curve.pdf", dpi=300)
    plt.clf()

    # plt.figure(figsize=(8, 6))
    # plt.plot(specificity_cnn_baseline, tpr_cnn_baseline, label=f'AUC (Baseline) = {auc_score_cnn_baseline:.2f}', color='green')
    # plt.plot(specificity_cnn, tpr_cnn, label=f'AUC (ours) = {auc_score_cnn:.2f}', color='blue')
    # plt.plot(specificity_lr, tpr_lr, label=f'AUC (Arnalich2019)= {auc_score_lr:.2f}', color='red')
    # # plt.plot(specificity_fus, tpr_fus, label=f'AUC (Fusion)= {auc_score_fus:.2f}', color='orange')
    # plt.xlabel('1 - Specificity (FPR)')
    # plt.ylabel('Sensibility (TPR)')
    # plt.title('Sensibility vs Specificity')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.clf()
