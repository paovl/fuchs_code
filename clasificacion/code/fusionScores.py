"""
@author: pvltarife
"""
import pickle
import numpy as np
import pdb
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import os

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
     with open(file_path, "a") as file:
        # Title
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
    
    # Read bagging score for best model CNN and for logistic regressor
    dict_standar_file = 'dict_scores_standar_bagging.pkl'
    data_dir1 = "results/RESNET/BIOMARKERS"
    data_dir2 = "results/RESNET/ELE_1_ELE_4_CORNEA-DENS_0_PAC_0"

    model1_dir = ['biomarkers']
    model2_dir = ['medical_rings3_angles3_bio_fusion5_50epochs']

    for i in np.arange(len(model1_dir)):
        save_scores_dir = f"{data_dir2}/{model2_dir[i]}/{'fusion_biomarkers'}"
        if not os.path.exists(save_scores_dir):
            os.makedirs(save_scores_dir)
        
        model1_standar_scores_path = f"{data_dir1}/{model1_dir[i]}/{'bagging'}/{dict_standar_file}"
        model2_standar_scores_path = f"{data_dir2}/{model2_dir[i]}/{'bagging'}/{dict_standar_file}"

        write_file = data_dir2 + '/' + model2_dir[i] + '/' + 'fusion_biomarkers.txt'

        print("\n SCORES for " + model1_dir[i] + '+' + model2_dir[i] + "\n\n")
        with open(write_file, "w") as file:
            file.write("\n SCORES for " + model1_dir[i] + '+' + model2_dir[i] + "\n\n")

        with open(model1_standar_scores_path, "rb") as f:
            scores1_standar = pickle.load(f)   

        with open(model2_standar_scores_path, "rb") as f:
            scores2_standar = pickle.load(f)  
    
        ids = scores1_standar['img_id']
        labels = scores2_standar['label']
        mean1 = np.mean(scores1_standar['score'])
        std1 = np.std(scores1_standar['score'])
        mean2 = np.mean(scores2_standar['score'])
        std2 = np.std(scores2_standar['score'])
        scores1_standar['score'] = (scores1_standar['score'] - mean1) / std1
        scores2_standar['score'] = (scores2_standar['score'] - mean2) / std2

        fusion_scores_standar = np.mean(np.column_stack((0.22*scores1_standar['score'], 0.78*scores2_standar['score'])), axis = 1)
        
        dict_variable = {'img_id': ids, 'scores 1': scores1_standar['score'] , 'scores 2': scores2_standar['score'], 'fusion_scores': fusion_scores_standar, 'label': labels}

        print_data(dict_variable)
        write_data(write_file, dict_variable)

        AUCs_standar= roc_auc_score(labels.astype(int), fusion_scores_standar)

        print(" AUC fusion: " + str(AUCs_standar))
        print("\n")
        with open(write_file, "a") as file:
            file.write("\n AUC fusion: " + str(AUCs_standar))
            file.write("\n")

        
