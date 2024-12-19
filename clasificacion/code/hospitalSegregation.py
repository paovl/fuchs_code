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
from EyeDataset import EyeDataset
import matplotlib.pyplot as plt

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
    dict_file = 'dict_scores_standar_bagging.pkl'
    data_dir = "results/RESNET/ELE_1_ELE_4_CORNEA-DENS_0_PAC_0"
    best_model_dir = "medical_rings3_angles3_fusion5_50epochs"
    
    if best_model_dir.find('bio') > 0:
        csvFile = '../datasets' + '/'+ 'dataset_global' + '/' + 'RESNET' + '/' + 'ransac_TH_1.5_r_45' + '/annotation_biomarkers.csv'
    else:
        csvFile = '../datasets' + '/'+ 'dataset_global' + '/' + 'RESNET' + '/' + 'ransac_TH_1.5_r_45' + '/annotation.csv'

    imageDir = '../datasets' + '/'+ 'dataset_global' + '/' + 'RESNET' + '/' + 'ransac_TH_1.5_r_45' + '/data'


    save_scores_dir = f"{data_dir}/{best_model_dir}/{'hospital_segregation'}"
    if not os.path.exists(save_scores_dir):
        os.makedirs(save_scores_dir)
    
    # Read bagging results for logits standarized
    best_model_scores_path = f"{data_dir}/{best_model_dir}/{'bagging'}/{dict_file}"
    write_file = data_dir+ '/' + best_model_dir + '/' + 'hospital_segregation.txt'

    print("HOSPITAL SEGREGATION for " + best_model_dir + "\n")
    with open(write_file, "w") as file:
        file.write("HOSPITAL SEGREGATION for " + best_model_dir + "\n")

    with open(best_model_scores_path, "rb") as f:
            scores = pickle.load(f)   
    
    full_dataset = EyeDataset(csvFile, imageDir,  test = False, random_seed = 0, testProp = 0)

    idxs = full_dataset.dataset.index
    hospitals = full_dataset.dataset['hospital']
    ids = scores['img_id']
    labels = scores['label']
    scores = scores['score']

    hospitals_names = full_dataset.dataset['hospital'].unique()
    AUCs_per_hospital = []
    scores_per_hospital= []
    labels_per_hospital = []
    ids_per_hospital = []
    n_images_per_hospital = []

    for hospital in hospitals_names:
        ids_hospital = ids[hospitals==hospital]
        Yhospital= labels[hospitals==hospital]
        scores_hospital= scores[hospitals==hospital]
        ids_per_hospital.append(ids_hospital)
        labels_per_hospital.append(Yhospital)
        scores_per_hospital.append(scores_hospital)
        n_images_per_hospital.append(len(ids_hospital))
        AUC = roc_auc_score(Yhospital, scores_hospital)
        AUCs_per_hospital.append(AUC)
        print(hospital + " n = " + str(len(ids_hospital)) + " AUC = " + str(AUC) + "\n")
        with open(write_file, "a") as file:
            file.write(hospital + " n = " + str(len(ids_hospital)) + " AUC = " + str(AUC) + "\n\n")
    
    # Save results
    hospital_segregation_scores = {'hospital': hospitals_names, 'img_id': ids_per_hospital, 'score': scores_per_hospital, 'label': labels_per_hospital}
    with open(save_scores_dir + '/hospital_segregation_scores.pkl', "wb") as f:
        pickle.dump(hospital_segregation_scores, f)
        
    hospital_segregation_AUCs = {'hospital': hospitals_names, 'AUCs': AUCs_per_hospital}
    with open(save_scores_dir + '/hospital_segregation_AUCs.pkl', "wb") as f:
        pickle.dump(hospital_segregation_AUCs, f)
    
    # Plot number of maps by hospital
    plt.figure(figsize=(10, 6))
    plt.bar(hospitals_names, n_images_per_hospital, color='blue')
    plt.xlabel('Hospitales')
    plt.ylabel('Cantidad de mapas')
    plt.title('Cantidad de mapas por hospital')
    plt.savefig(save_scores_dir + '/nmaps.png')
        
