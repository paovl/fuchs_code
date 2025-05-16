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
from sklearn.metrics import roc_curve

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
    data_dir = "results/SEP/ELE_1_ELE_4_CORNEA-DENS_0_PAC_0/medical_rings3_angles3_fusion5_50epochs"
    bio_data_dir = "results/SEP/BIOMARKERS/biomarkers"

    model_dir = ['no_deactivate_seed0_dseed0_th0.07_a1000_sigmoidWeightedLoss(full DB)_bio']

    bio_model_standar_scores_path = f"{bio_data_dir}/{'bagging'}/{dict_standar_file}"
    with open(bio_model_standar_scores_path, "rb") as f:
        bio_scores= pickle.load(f)
    
    bio_ids = bio_scores['img_id']
    bio_labels = bio_scores['label']
    bio_scores = bio_scores['score']

    bio_scores_norm = 1 / (1 + np.exp(-bio_scores))
    # bio_scores_norm =np.exp(bio_scores)

    bio_fpr, bio_tpr, bio_thresholds = roc_curve(bio_labels, bio_scores_norm)
    bio_specificity = 1 - bio_fpr

    bio_predict_labels = (bio_scores_norm >= 0.5)
    bio_recall = metrics.recall_score(bio_labels, bio_predict_labels)
    bio_specifity = metrics.recall_score(bio_labels, bio_predict_labels, pos_label = 0)
    bio_precision = metrics.precision_score(bio_labels, bio_predict_labels)
    bio_false_positive_rate = 1 - bio_specifity
    bio_roc_auc = roc_auc_score(bio_labels, bio_scores_norm)
    bio_f1 = metrics.f1_score(bio_labels, bio_predict_labels)

    print("\nBIO THRESHOLD 0.5\n")
    print("PRECISION = {}".format(bio_precision))
    print("RECALL = {}".format(bio_recall))
    print("SPECIFICITY = {}".format(bio_specifity))
    print("FPR = {}".format(bio_false_positive_rate))
    print("AUC = {}".format(bio_roc_auc))
    print("F1 = {}".format(bio_f1))

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
        # scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        scores_norm = 1 / (1 + np.exp(-scores))
        predict_labels = (scores_norm >= 0.5)
        recall = metrics.recall_score(labels, predict_labels)
        specifity = metrics.recall_score(labels, predict_labels, pos_label = 0)
        precision = metrics.precision_score(labels, predict_labels)
        false_positive_rate = 1 - specifity
        roc_auc = roc_auc_score(labels, scores_norm)
        f1 = metrics.f1_score(labels, predict_labels)

        print("\nTHRESHOLD 0.5\n")
        print("PRECISION = {}".format(precision))
        print("RECALL = {}".format(recall))
        print("SPECIFICITY = {}".format(specifity))
        print("FPR = {}".format(false_positive_rate))
        print("AUC = {}".format(roc_auc))
        print("F1 = {}".format(f1))
        with open(write_file, "w") as file:
            file.write("\nTHRESHOLD 0.5\n\n")
            file.write("PRECISION = {}".format(precision))
            file.write("RECALL = {}".format(recall))
            file.write("SPECIFITY = {}".format(specifity))
            file.write("FPR = {}".format(false_positive_rate))
            file.write("AUC = {}".format(roc_auc))
            file.write("F1 = {}".format(f1))


        # Metrics for recall = 95%
        precisions, recalls, thresholds = precision_recall_curve(labels, scores_norm)
        threshold_95 =  thresholds[np.argmin(recalls >= 0.95)-1]

        predict_labels_95 = (scores_norm >= threshold_95).astype(int)
        predict_labels_95 = predict_labels_95.astype(int)
        
        precision_95 = metrics.precision_score(labels, predict_labels_95)
        recall_95 = metrics.recall_score(labels, predict_labels_95) # 
        specifity_95 = metrics.recall_score(labels, predict_labels_95, pos_label=0) # 
        false_positive_rate_95 = 1 - specifity_95
        roc_auc_95 = roc_auc_score(labels, predict_labels_95)
        f1_95 = metrics.f1_score(labels, predict_labels_95)

        print("\nRECALL AT 95%")
        print("\nTHRESHOLD (95%) = {}".format(threshold_95))
        print("PRECISION (95%) = {}".format(precision_95))
        print("RECALL (95%) = {}".format(recall_95))
        print("SPECIFITY (95%) = {}".format(specifity_95))
        print("FPR (95%) = {}".format(false_positive_rate_95))
        print("AUC (95%) = {}".format(roc_auc_95))
        print("F1 (95%) = {}".format(f1_95))

        with open(write_file, "w") as file:
            file.write("\nRECALL AT 95%\n")
            file.write("\n\nTHRESHOLD (95%) = {}".format(threshold_95))
            file.write("\nPRECISION (95%) = {}".format(precision_95))
            file.write("\nRECALL (95%) = {}".format(recall_95))
            file.write("\nSPECIFITY (95%) = {}".format(specifity_95))
            file.write("\nFPR (95%) = {}".format(false_positive_rate_95))
            file.write("AUC (95%) = {}".format(roc_auc_95))
            file.write("\nF1 (95%) = {}".format(f1_95))
        
        specifities = []
        AUCs = []
        # Specifity - Recall curve 
        for th in thresholds:
            predict_labels_aux = (scores_norm >= th)
            specifities.append(metrics.recall_score(labels, predict_labels_aux, pos_label=0))
            AUCs.append(roc_auc_score(labels, predict_labels_aux))
        
        plt.figure(1)
        plt.plot(thresholds, specifities, "b--", label="Specificity")
        plt.plot(thresholds, recalls[:-1], "g--", label="Sensitivity")
        plt.xlabel("Threshold")
        plt.legend()
        plt.title( "Specificity - Sensitivity")
        plt.savefig(save_scores_dir + "/specifity_sensitivity.png")
        plt.savefig(save_scores_dir + "/specifity_sensitivity.pdf")

        # Create a confusion matrix using sklearn
        
        plt.figure(2)
        conf_matrix = confusion_matrix(labels, predict_labels) 
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

        #print ids and scores and labels --> false negatives with thereshold 0.5
        false_negatives = {'img_id': [], 'score': [], 'label': []}
        for i in range(len(ids)):
            if labels[i] == 1 and predict_labels[i] == 0:
                false_negatives['img_id'].append(ids[i])
                false_negatives['score'].append(scores_norm[i])
                false_negatives['label'].append(labels[i])
        print("\nFALSE NEGATIVES (THRESHOLD 0.5)")
        print_data(false_negatives)
        write_data(save_scores_dir + "/false_negatives_0.5.txt", false_negatives)

        #print ids and scores and labels --> false positives with thereshold 0.5
        false_positives = {'img_id': [], 'score': [], 'label': []}
        for i in range(len(ids)):
            if labels[i] == 0 and predict_labels[i] == 1:
                false_positives['img_id'].append(ids[i])
                false_positives['score'].append(scores_norm[i])
                false_positives['label'].append(labels[i])
        print("\nFALSE POSITIVES (THRESHOLD 0.5)")
        print_data(false_positives)
        write_data(save_scores_dir + "/false_positives_0.5.txt", false_positives)

        # print ids and scores and labels --> true positives with thereshold 0.5
        true_positives = {'img_id': [], 'score': [], 'label': []}
        for i in range(len(ids)):
            if labels[i] == 1 and predict_labels[i] == 1:
                true_positives['img_id'].append(ids[i])
                true_positives['score'].append(scores_norm[i])
                true_positives['label'].append(labels[i])
        print("\nTRUE POSITIVES (THRESHOLD 0.5)")
        print_data(true_positives)

        # print ids and scores and labels --> true negatives with thereshold 0.5
        true_negatives = {'img_id': [], 'score': [], 'label': []}
        for i in range(len(ids)):
            if labels[i] == 0 and predict_labels[i] == 0:
                true_negatives['img_id'].append(ids[i])
                true_negatives['score'].append(scores_norm[i])
                true_negatives['label'].append(labels[i])
        print("\nTRUE NEGATIVES (THRESHOLD 0.5)")
        print_data(true_negatives)

        image_fpr, image_tpr, image_thresholds = roc_curve(labels, scores_norm)
        image_specificity = 1 - image_fpr

        plt.figure(4)
        plt.plot(image_specificity, image_tpr, 'orange')
        plt.plot(bio_specificity, bio_tpr, 'r')
        plt.xlabel('Specificity', fontsize=12)
        plt.ylabel('Sensitivity', fontsize=12)
        plt.title('ROC curve (Sensitivity - Specificity)', fontsize=14)
        plt.legend(['CNN', 'Biomarkers'], fontsize=12)
        plt.grid()
        plt.savefig(save_scores_dir + "/roc_specifity_sensitivity.png")
        plt.savefig(save_scores_dir + "/roc_specifity_sensitivity.pdf")
        plt.show()
                
