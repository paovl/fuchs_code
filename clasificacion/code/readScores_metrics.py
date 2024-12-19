"""
@author: pvltarife
"""
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve, confusion_matrix

import pickle
import numpy as np
import pdb
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import seaborn as sns
import copy

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

def write_data(file_path, data):
    # Title
     with open(file_path, "a") as file:
        # Encabezado
        file.write('{:<12}'.format('img_id'))
        for key in data.keys():
            if key != 'img_id':
                file.write('{:<12}'.format(key))
        file.write('\n')
    
        # Data
        for i in range(len(data['img_id'])):
            file.write('{:<12}'.format(data['img_id'][i]))
            for key in data.keys():
                if key != 'img_id':
                    file.write('{:<12.8f}'.format(data[key][i]))
            file.write('\n')
    

if '__main__':

    # Read bagging scores from the best model
    dict_standar_file = 'dict_scores_standar_bagging.pkl'
    data_dir = "results/RESNET/ELE_1_ELE_4_CORNEA-DENS_0_PAC_0"

    model_dir = ['medical_rings3_angles3_fusion5_50epochs']

    for dir in model_dir:
        save_scores_dir = f"{data_dir}/{dir}/{'metrics'}"
        if not os.path.exists(save_scores_dir):
            os.makedirs(save_scores_dir)
        
        model_standar_scores_path = f"{data_dir}/{dir}/{'bagging'}/{dict_standar_file}"
        write_file = data_dir + '/' + dir + '/' + 'metrics.txt'

        print("\n METRICS for " + dir + "\n\n")
        with open(write_file, "w") as file:
            file.write("\n METRICS for " + dir + "\n\n")

        with open(model_standar_scores_path, "rb") as f:
            scores= pickle.load(f)   

        ids = scores['img_id']
        labels = scores['label']
        scores = scores['score']

        # Metrics in general for thereshold 0.5
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        predict_labels = (scores >= 0.5)
        recall = metrics.recall_score(labels, predict_labels)
        specifity = metrics.recall_score(labels, predict_labels, pos_label = 0)
        precision = metrics.precision_score(labels, predict_labels)
        false_positive_rate = 1 - specifity
        roc_auc = roc_auc_score(labels, scores_norm)

        print("\nTHRESHOLD 0.5\n")
        print("PRECISION = {}".format(precision))
        print("RECALL = {}".format(recall))
        print("SPECIFITY = {}".format(specifity))
        print("FPR = {}".format(false_positive_rate))
        print("AUC = {}".format(roc_auc))
        with open(write_file, "w") as file:
            file.write("\nTHRESHOLD 0.5\n\n")
            file.write("PRECISION = {}".format(precision))
            file.write("RECALL = {}".format(recall))
            file.write("SPECIFITY = {}".format(specifity))
            file.write("FPR = {}".format(false_positive_rate))
            file.write("AUC = {}".format(roc_auc))

        # Metrics for recall = 95%
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        threshold_95 =  thresholds[np.argmin(recalls >= 0.95)]

        predict_labels_95 = (scores >= threshold_95)
        predict_labels_95 = predict_labels_95.astype(int)
        
        precision_95 = metrics.precision_score(labels, predict_labels_95)
        recall_95 = metrics.recall_score(labels, predict_labels_95) # 
        specifity_95 = metrics.recall_score(labels, predict_labels_95, pos_label=0) # 
        false_positive_rate_95 = 1 - specifity_95
        roc_auc_95 = roc_auc_score(labels, predict_labels_95)

        print("\nRECALL AT 95%")
        print("\nTHRESHOLD (95%) = {}".format(threshold_95))
        print("PRECISION (95%) = {}".format(precision_95))
        print("RECALL (95%) = {}".format(recall_95))
        print("SPECIFITY (95%) = {}".format(specifity_95))
        print("FPR (95%) = {}".format(false_positive_rate_95))
        print("AUC (95%) = {}".format(roc_auc_95))
        with open(write_file, "w") as file:
            file.write("\nRECALL AT 95%\n")
            file.write("\n\nTHRESHOLD (95%) = {}".format(threshold_95))
            file.write("\nPRECISION (95%) = {}".format(precision_95))
            file.write("\nRECALL (95%) = {}".format(recall_95))
            file.write("\nSPECIFITY (95%) = {}".format(specifity_95))
            file.write("\nFPR (95%) = {}".format(false_positive_rate_95))
            file.write("AUC (95%) = {}".format(roc_auc_95))
        
        specifities = []
        AUCs = []
        # Specifity - Recall curve 
        for th in thresholds:
            predict_labels_aux = (scores >= th)
            specifities.append(metrics.recall_score(labels, predict_labels_aux, pos_label=0))
            AUCs.append(roc_auc_score(labels, predict_labels_aux))
        
        plt.figure(1)
        plt.plot(thresholds, specifities, "b--", label="Especificidad")
        plt.plot(thresholds, recalls[:-1], "g--", label="Sensibilidad")
        plt.xlabel("Umbral")
        plt.legend()
        plt.title( "Especificidad - Sensibilidad")
        plt.savefig(save_scores_dir + "/specifity_recall.png")

        # Create a confusion matrix using sklearn
        
        plt.figure(2)
        conf_matrix = confusion_matrix(labels, predict_labels_95) 
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [0, 1],)
        ax = cm_display.plot()
        plt.title("Matríz de confusión")
        plt.xlabel("Etiquetas predichas")
        plt.ylabel("Etiquetas reales")
        plt.savefig(save_scores_dir + "/confusion_matrix.png")
        fpr, tpr, threshold = metrics.roc_curve(labels, predict_labels_95)

        plt.figure(3)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc_95)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate (TPR)')
        plt.xlabel('False Positive Rate (TPR)')
        plt.savefig(save_scores_dir + "/roc_auc_curve.png")
        
