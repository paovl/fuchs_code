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
    
    # Title
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

    # Read logits by iteration
    dict_raw_file = 'dict_raw_scores.pkl'
    dict_file = 'dict_scores.pkl'
    scores_file = 'out_scores.pkl'
    data_dir = "results/RESNET/ELE_1_ELE_4_CORNEA-DENS_0_PAC_0"

    model1_dir = ['baseline_b_50epochs', 'baseline_bio_50epochs']
    model2_dir = ['medical_rings3_angles3_50epochs', 'medical_rings3_angles3_bio_50epochs']

    for i in np.arange(len(model1_dir)):
        if model1_dir[i].find('bio') > 0:
            test_idxs_path = data_dir + '/test_idxs_by_iter_BIO.pkl'
        else:
            test_idxs_path = data_dir + '/test_idxs_by_iter.pkl'
        
        with open(test_idxs_path, "rb") as f:
            test_idxs = pickle.load(f)   

        save_scores_dir = f"{data_dir}/{model2_dir[i]}/{'ensemble'}"
        if not os.path.exists(save_scores_dir):
            os.makedirs(save_scores_dir)
        
        model1_scores_path = f"{data_dir}/{model1_dir[i]}/{'results_nn'}/{scores_file}"
        model2_scores_path = f"{data_dir}/{model2_dir[i]}/{'results_nn'}/{scores_file}"

        dict1_scores_path = f"{data_dir}/{model1_dir[i]}/{'results_nn'}/{dict_file}"
        dict2_scores_path = f"{data_dir}/{model2_dir[i]}/{'results_nn'}/{dict_file}"

        dict_raw_scores_path = f"{data_dir}/{model2_dir[i]}/{'results_nn'}/{dict_raw_file}"

        write_file = data_dir+ '/' + model2_dir[i] + '/' + 'ensemble.txt'

        print("\nSCORES for " + model1_dir[i] + '+' + model2_dir[i] + "\n\n")
        with open(write_file, "w") as file:
            file.write("\SCORES for " + model1_dir[i] + '+' + model2_dir[i] + "\n\n")

        with open(model1_scores_path, "rb") as f:
            scores1 = pickle.load(f)   

        with open(model2_scores_path, "rb") as f:
            scores2 = pickle.load(f)  
    
        with open(dict1_scores_path, "rb") as f:
            dict1 = pickle.load(f)   

        with open(dict2_scores_path, "rb") as f:
            dict2 = pickle.load(f) 
        
        with open(dict_raw_scores_path, "rb") as f:
            dict_raw_scores = pickle.load(f)  
    
        ids = dict1['img_id']
        labels = dict1['label']

        iterations = len(scores1)
        AUCs = np.zeros((iterations,))
        total_scores = []
        AUCs_standar = np.zeros((iterations,))
        total_scores_standar =[]
        total_scores_org = np.full(dict_raw_scores['score'].shape, np.nan)
        total_scores_standar_org = np.full(dict_raw_scores['score'].shape, np.nan)

        for j in np.arange(iterations):
            # Show in screen
            print("ITERATION {} \n".format(j))

            # Save in txt file
            with open(write_file, "a") as file:
                file.write("ITERATION " + str(j) + " \n")

            fusion_scores = np.mean(np.column_stack((scores1[j], scores2[j])), axis = 1)
                                    
            scores1_standar = (scores1[j] - np.mean(scores1[j])) / np.std(scores1[j])
            scores2_standar = (scores2[j] - np.mean(scores2[j])) / np.std(scores2[j])
            fusion_scores_standar = np.mean(np.column_stack((scores1_standar, scores2_standar)), axis = 1)

            dict_variable = {'img_id': ids[j], 'scores 1': scores1[j] , 'scores 2': scores2[j], 'fusion_scores': fusion_scores, 'label': labels[j]}

            print_data(dict_variable)
            write_data(write_file, dict_variable)

            # Calculate AUC
            total_scores.append(fusion_scores)
            total_scores_standar.append(fusion_scores_standar)
            total_scores_org[test_idxs[j], j] = fusion_scores
            total_scores_standar_org[test_idxs[j], j] = fusion_scores_standar

            AUCs[j]= roc_auc_score(labels[j].astype(int), fusion_scores)
            AUCs_standar[j]= roc_auc_score(labels[j].astype(int), fusion_scores_standar)

            print("AUC : " + str(AUCs[j]))
            print("AUC standarization: " + str(AUCs_standar[j]))
            print("\n")
            with open(write_file, "a") as file:
                file.write("AUC: " + str(AUCs[j]) + '\n')
                file.write("AUC standarization: " + str(AUCs_standar[j])+ '\n\n')

        print("Mean AUC across iterations: " + str(np.mean(AUCs)))
        print("Mean AUC standar across iterations: " + str(np.mean(AUCs_standar)))
        print("\n")
        with open(write_file, "a") as file:
            file.write("\n")
            file.write("Mean AUC across iterations: " + str(np.mean(AUCs)))
            file.write("\nMean AUC standar across iterations: " + str(np.mean(AUCs_standar)))

        dict_scores_raw_ensemble = {'img_id': ids, 'score': total_scores, 'label': labels}
        with open(save_scores_dir + '/dict_scores_raw_ensemble.pkl', "wb") as f:
            pickle.dump(dict_scores_raw_ensemble, f)
        
        dict_scores_standar_ensemble = {'img_id': ids, 'score': total_scores_standar, 'label': labels}
        with open(save_scores_dir + '/dict_scores_standar_ensemble.pkl', "wb") as f:
            pickle.dump(dict_scores_standar_ensemble, f)
        
        dict_scores_matrix_raw_ensemble = {'img_id': dict_raw_scores['img_id'], 'score': total_scores_org, 'label': dict_raw_scores['label']}
        with open(save_scores_dir + '/dict_scores_matrix_raw_ensemble.pkl', "wb") as f:
            pickle.dump(dict_scores_matrix_raw_ensemble, f)
        
        dict_scores_matrix_standar_ensemble = {'img_id': dict_raw_scores['img_id'], 'score': total_scores_standar_org, 'label': dict_raw_scores['label']}
        with open(save_scores_dir + '/dict_scores_matrix_standar_ensemble.pkl', "wb") as f:
            pickle.dump(dict_scores_matrix_standar_ensemble, f)
        

        
