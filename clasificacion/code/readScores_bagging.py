"""
@author: pvltarife
"""
import pickle
import numpy as np
import pdb
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import pandas as pd
import copy
import os
import sys

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
    
def standar_maxmin_scores (scores_matrix, raw):
    if raw == 0 :
        device  = torch.device("cuda:0")
        scores_matrix = torch.sigmoid(torch.tensor(scores_matrix).to(device))
        scores_matrix = scores_matrix.cpu().numpy()

    mean_scores_rows = np.full((scores_matrix.shape[0],), np.nan)
    std_scores_rows = np.full((scores_matrix.shape[0],), np.nan)
    mean_scores_rows_standar = np.full((scores_matrix.shape[0],), np.nan)
    mean_scores_rows_maxmin = np.full((scores_matrix.shape[0],), np.nan)

    mean_scores_columns = np.full((scores_matrix.shape[1],), np.nan)
    std_scores_columns = np.full((scores_matrix.shape[1],), np.nan)
    mean_scores_columns_standar = np.full((scores_matrix.shape[0],), np.nan)
    mean_scores_columns_maxmin = np.full((scores_matrix.shape[0],), np.nan)

    # noNan values
    idx_Nan = np.where(np.isnan(scores_matrix))
    
    # matrix with zeros where nan
    scores_matrix_noNan_rows = np.nan_to_num(scores_matrix, nan=0)
    scores_matrix_noNan_columns = np.nan_to_num(scores_matrix, nan=0)

    # Rows
    sum_noNan_rows = np.sum(~np.isnan(scores_matrix), axis=1)
    valid_sum_noNan_rows = np.where(sum_noNan_rows > 0)

    # mean by rows
    mean_scores_rows[valid_sum_noNan_rows] = np.sum(scores_matrix_noNan_rows[valid_sum_noNan_rows,:][0], axis=1) / sum_noNan_rows[valid_sum_noNan_rows]
    # std by rows
    scores_matrix_noNan_rows_std = (scores_matrix_noNan_rows - mean_scores_rows[:, np.newaxis])**2
    scores_matrix_noNan_rows_std[idx_Nan] = 0
    std_scores_rows[valid_sum_noNan_rows] = np.sqrt(np.sum(scores_matrix_noNan_rows_std[valid_sum_noNan_rows,:][0], axis=1) / sum_noNan_rows[valid_sum_noNan_rows])

    scores_matrix_noNan_rows_standar = (scores_matrix_noNan_rows - mean_scores_rows[:, np.newaxis]) / std_scores_rows[:, np.newaxis]
    scores_matrix_noNan_rows_standar[idx_Nan] = 0

    # max & min by rows
    max_scores_rows = np.max(scores_matrix_noNan_rows_standar, axis=1)
    min_scores_rows = np.min(scores_matrix_noNan_rows_standar, axis=1)

    scores_matrix_noNan_rows_maxmin = (scores_matrix_noNan_rows - min_scores_rows[:, np.newaxis]) / (max_scores_rows[:, np.newaxis] - min_scores_rows[:, np.newaxis])
    scores_matrix_noNan_rows_maxmin[idx_Nan] = 0

    # mean scores by rows
    mean_scores_rows_standar[valid_sum_noNan_rows] = np.sum(scores_matrix_noNan_rows_standar[valid_sum_noNan_rows,:][0], axis=1) / sum_noNan_rows[valid_sum_noNan_rows]
    mean_scores_rows_maxmin[valid_sum_noNan_rows] = np.sum(scores_matrix_noNan_rows_maxmin[valid_sum_noNan_rows,:][0], axis=1) / sum_noNan_rows[valid_sum_noNan_rows]
    
    # Columns
    sum_noNan_columns = np.sum(~np.isnan(scores_matrix), axis=0)
    valid_sum_noNan_columns = np.where(sum_noNan_columns > 0)

    # mean by columns
    mean_scores_columns[valid_sum_noNan_columns] = (np.sum(scores_matrix_noNan_columns, axis=0) / sum_noNan_columns)
    # std by columns
    scores_matrix_noNan_columns_std = (scores_matrix_noNan_columns - mean_scores_columns[:,np.newaxis].T)**2
    scores_matrix_noNan_columns_std[idx_Nan] = 0
    std_scores_columns[valid_sum_noNan_columns] = np.sqrt(np.sum(scores_matrix_noNan_columns_std, axis=0) / sum_noNan_columns)

    scores_matrix_noNan_columns_standar = (scores_matrix_noNan_columns - mean_scores_columns[:, np.newaxis].T) / std_scores_columns[:, np.newaxis].T
    scores_matrix_noNan_columns_standar[idx_Nan] = 0

    # max & min by columns
    max_scores_columns = np.max(scores_matrix_noNan_columns_standar, axis=0)
    min_scores_columns = np.min(scores_matrix_noNan_columns_standar, axis=0)

    scores_matrix_noNan_columns_maxmin = (scores_matrix_noNan_columns - min_scores_columns[:, np.newaxis].T) / (max_scores_columns[:, np.newaxis].T - min_scores_columns[:, np.newaxis].T)
    scores_matrix_noNan_columns_maxmin[idx_Nan] = 0

    # mean scores by rows
    mean_scores_columns_standar[valid_sum_noNan_rows] = np.sum(scores_matrix_noNan_columns_standar[valid_sum_noNan_rows], axis=1) / sum_noNan_rows[valid_sum_noNan_rows]
    mean_scores_columns_maxmin[valid_sum_noNan_rows] = np.sum(scores_matrix_noNan_columns_maxmin[valid_sum_noNan_rows], axis=1) / sum_noNan_rows[valid_sum_noNan_rows]
    return mean_scores_rows, mean_scores_rows_standar, mean_scores_rows_maxmin, mean_scores_columns_standar, mean_scores_columns_maxmin, valid_sum_noNan_rows

if '__main__':
    """ Arguments
        dir: results directory
    """
    eConfig = {
        'dir':'SEP',
        'dir_results':'no_deactivate_seed0_dseed0_th0.07_a1000_sigmoidWeightedLoss(full DB)_70train'
        }
    
    args = sys.argv[1::]
    for i in range(0,len(args),2):
        key = args[i]
        val = args[i+1]
        eConfig[key] = type(eConfig[key])(val)
        print (str(eConfig[key]))
          
    # Read scores matrix with dimensions KxN. K = dataset length, N = number of iterations (70)
    #Generate bagging scores for CNN
    scores_files = ['dict_raw_scores.pkl', 'dict_scores_matrix.pkl']
    data_dirs = ["results/" + eConfig['dir'] + "/ELE_1_ELE_4_CORNEA-DENS_0_PAC_0"]
    input_dirs = ['medical_rings3_angles3_fusion5_50epochs/'  + eConfig['dir_results']]
    
    # Uncomment if you want to generate bagging scores for logistic regressor
    
    # Generate bagging scores for logistic regressor
    # scores_files = ['dict_scores_matrix.pkl']
    # data_dirs = ["results/SEP/BIOMARKERS"]
    # input_dirs = ["biomarkers"]
    
    for i in np.arange(len(data_dirs)):

        input_dir = input_dirs[i]
        scores_file = scores_files[i]
        data_dir = data_dirs[i]

        save_scores_dir = f"{data_dir}/{input_dir}/{'bagging'}"

        if not os.path.exists(save_scores_dir):
            os.makedirs(save_scores_dir)

        scores_path = f"{data_dir}/{input_dir}/{'results_nn'}/{scores_file}"
        write_file = data_dir + '/' + input_dir + '/' + 'bagging.txt'

        print("\nSCORES for " + input_dir + " MAPS")
        with open(write_file, "w") as file:
            file.write("SCORES for " + input_dir + " MAPS\n")
        
        with open(scores_path, "rb") as f:
            scores_dict = pickle.load(f)
        
        ids_list = scores_dict['img_id']
        scores_matrix = scores_dict['score']
        labels_list = scores_dict['label']
        iterations = scores_matrix.shape[1]

        # RAW SCORES
        mean_scores_rows, mean_scores_rows_standar, mean_scores_rows_maxmin, mean_scores_columns_standar, mean_scores_columns_maxmin, valid_sum_noNan_rows = standar_maxmin_scores(scores_matrix, 1)
        # SIGMOID SCORES
        mean_scores_rows_sigmoid, mean_scores_rows_standar_sigmoid, mean_scores_rows_maxmin_sigmoid, mean_scores_columns_standar_sigmoid, mean_scores_columns_maxmin_sigmoid,valid_sum_noNan_rows= standar_maxmin_scores(scores_matrix, 0)

        device  = torch.device("cuda:0")

        scores_norm = 1 / (1 + np.exp(-mean_scores_columns_standar))
        dict_variable = {'img_id': ids_list, 'score': scores_norm, 'label': labels_list}

        # print norm scores for standardized logits
        print_data(dict_variable)
        write_data(write_file, dict_variable)

        test_labels = dict_variable['label']

        valid_sum_noNan_rs = valid_sum_noNan_rows[0]

        AUCs= roc_auc_score(test_labels[valid_sum_noNan_rows].astype(int), mean_scores_rows[valid_sum_noNan_rows])
        #AUCs_rows_standar= roc_auc_score(test_labels[valid_sum_noNan_rows].astype(int), mean_scores_rows_standar[valid_sum_noNan_rows])
        #AUCs_rows_maxmin= roc_auc_score(test_labels[valid_sum_noNan_rows].astype(int), mean_scores_rows_maxmin[valid_sum_noNan_rows])
        AUCs_columns_standar= roc_auc_score(test_labels[valid_sum_noNan_rows].astype(int), mean_scores_columns_standar[valid_sum_noNan_rows])
        AUCs_columns_maxmin= roc_auc_score(test_labels[valid_sum_noNan_rows].astype(int), mean_scores_columns_maxmin[valid_sum_noNan_rows])
        
        AUCs_sigmoid = roc_auc_score(test_labels[valid_sum_noNan_rows].astype(int), mean_scores_rows_sigmoid[valid_sum_noNan_rows])
        #AUCs_rows_standar_sigmoid = roc_auc_score(test_labels[valid_sum_noNan_rows].astype(int), mean_scores_rows_standar_sigmoid[valid_sum_noNan_rows])
        #AUCs_rows_maxmin_sigmoid = roc_auc_score(test_labels[valid_sum_noNan_rows].astype(int), mean_scores_rows_maxmin_sigmoid[valid_sum_noNan_rows])
        AUCs_columns_standar_sigmoid = roc_auc_score(test_labels[valid_sum_noNan_rows].astype(int), mean_scores_columns_standar_sigmoid[valid_sum_noNan_rows])
        AUCs_columns_maxmin_sigmoid = roc_auc_score(test_labels[valid_sum_noNan_rows].astype(int), mean_scores_columns_maxmin_sigmoid[valid_sum_noNan_rows])

        print("\nRAW SCORES ")
        print("AUC : " + str(AUCs))
        #print(" AUC rows standar : " + str(AUCs_rows_standar))
        #print(" AUC rows maxmin : " + str(AUCs_rows_maxmin))
        print("AUC standarization : " + str(AUCs_columns_standar))
        print("AUC norm maxmin : " + str(AUCs_columns_maxmin))
        # Show in screen
        print("\nSIGMOID SCORES ")
        print("AUC sigmoid: " + str(AUCs))
        #print(" AUC rows standar sigmoid: " + str(AUCs_rows_standar_sigmoid))
        #print(" AUC rows maxmin sigmoid: " + str(AUCs_rows_maxmin_sigmoid))
        print("AUC standarization sigmoid: " + str(AUCs_columns_standar_sigmoid))
        print("AUC norm maxmin sigmoid: " + str(AUCs_columns_maxmin_sigmoid))
        with open(write_file, "a") as file:
            file.write("\n")
            # Show in screen
            file.write("\n\nRAW SCORES ")
            file.write("\nAUC : " + str(AUCs))
            #file.write("\n AUC rows standar : " + str(AUCs_rows_standar))
            #file.write("\n AUC rows maxmin : " + str(AUCs_rows_maxmin))
            file.write("\nAUC standarization : " + str(AUCs_columns_standar))
            file.write("\nAUC norm maxmin : " + str(AUCs_columns_maxmin))
            # Show in screen
            file.write("\n\nSIGMOID SCORES ")
            file.write("\nAUC sigmoid: " + str(AUCs))
            #file.write("\n AUC rows standar sigmoid: " + str(AUCs_rows_standar_sigmoid))
            #file.wriowte("\n AUC rows maxmin sigmoid: " + str(AUCs_rows_maxmin_sigmoid))
            file.write("\nAUC standarization sigmoid: " + str(AUCs_columns_standar_sigmoid))
            file.write("\nAUC norm maxmin sigmoid: " + str(AUCs_columns_maxmin_sigmoid))
        
        # Save all type of scores
        
        dict_scores_raw_bagging = {'img_id': ids_list, 'score': mean_scores_rows, 'label': labels_list}
        with open(save_scores_dir + '/dict_scores_raw_bagging.pkl', "wb") as f:
            pickle.dump(dict_scores_raw_bagging, f)
        
        dict_scores_standar_bagging = {'img_id': ids_list, 'score': mean_scores_columns_standar, 'label': labels_list}
        with open(save_scores_dir+ '/dict_scores_standar_bagging.pkl', "wb") as f:
            pickle.dump(dict_scores_standar_bagging, f)
        
        dict_scores_maxmin_bagging = {'img_id': ids_list, 'score': mean_scores_columns_maxmin, 'label': labels_list}
        with open(save_scores_dir+ '/dict_scores_maxmin_bagging.pkl', "wb") as f:
            pickle.dump(dict_scores_maxmin_bagging, f)
        
        dict_scores_sigmoid_bagging = {'img_id': ids_list, 'score': mean_scores_rows_sigmoid, 'label': labels_list}
        with open(save_scores_dir+ '/dict_scores_sigmoid_bagging.pkl', "wb") as f:
            pickle.dump(dict_scores_sigmoid_bagging, f)
        
        dict_scores_standar_sigmoid_bagging = {'img_id': ids_list, 'score': mean_scores_columns_standar_sigmoid, 'label': labels_list}
        with open(save_scores_dir+ '/dict_scores_standar_sigmoid_bagging.pkl', "wb") as f:
            pickle.dump(dict_scores_standar_sigmoid_bagging, f)
        
        dict_scores_maxmin_sigmoid_bagging = {'img_id': ids_list, 'score': mean_scores_columns_maxmin_sigmoid, 'label': labels_list}
        with open(save_scores_dir+ '/dict_scores_maxmin_sigmoid_bagging.pkl', "wb") as f:
            pickle.dump(save_scores_dir+ '/dict_scores_maxmin_sigmoid_bagging.pkl', f)

