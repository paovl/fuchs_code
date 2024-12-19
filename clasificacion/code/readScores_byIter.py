import pickle
import numpy as np
import pdb
from sklearn.metrics import roc_auc_score
import torch

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
    # Read scores dict by ITERATIONS
    scores_file = 'dict_scores.pkl'
    data_dir = "results/RESNET"
    dir = 'ELE_1_ELE_4_CORNEA-DENS_0_PAC_0'
    input_dir = 'medical_rings3_angles3_fusion5_50epochs'
    
    scores_path = f"{data_dir}/{dir}/{input_dir}/{'results_nn'}/{scores_file}"
    write_file = data_dir + '/' + dir + '/' + input_dir + '/scores_out_results.txt'
 
    print("SCORES for " + dir + " MAPS\n\n")
    with open(write_file, "w") as file:
        file.write("SCORES for " + dir + " MAPS\n\n")
        
    with open(scores_path, "rb") as f:
        scores_dict = pickle.load(f)
    
    
    ids_list = scores_dict['img_id']
    scores_list = scores_dict['score']
    labels_list = scores_dict['label']
    iterations = len(ids_list)
    AUCs = np.zeros((iterations,))
    device  = torch.device("cuda:0")

    for j in np.arange(len(ids_list)):

        print("ITERATION {} \n".format(j))
        with open(write_file, "a") as file:
            file.write("\n\nITERATION " + str(j) + " \n")

        dict_variable = {'img_id': ids_list[j], 'score': scores_list[j], 'label': labels_list[j]}

        print_data(dict_variable)
        write_data(write_file, dict_variable)

        # Calculate AUC
        test_scores = dict_variable['score']
        test_labels = dict_variable['label']

        AUCs[j]= roc_auc_score(test_labels.astype(int), test_scores)


        print("AUC : " + str(AUCs[j]))
        print("\n")
        with open(write_file, "a") as file:
            file.write("\n")
            file.write("AUC: " + str(AUCs[j]))

    print("Mean AUC across iterations: " + str(np.mean(AUCs)))
    print("\n")

    with open(write_file, "a") as file:
        file.write("\n\n")
        file.write("Mean AUC across iterations: " + str(np.mean(AUCs)))