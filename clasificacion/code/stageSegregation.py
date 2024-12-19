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
    
    save_scores_dir = f"{data_dir}/{best_model_dir}/{'stage_segregation'}"
    if not os.path.exists(save_scores_dir):
        os.makedirs(save_scores_dir)
    
    # Read bagging results for standardized logits
    best_model_scores_path = f"{data_dir}/{best_model_dir}/{'bagging'}/{dict_file}"
    write_file = data_dir+ '/' + best_model_dir + '/' + 'stage_segregation.txt'
    with open(best_model_scores_path, "rb") as f:
            scores = pickle.load(f)   

    stage_path = data_dir + '/dict_stages_global.pkl'
    with open(stage_path, "rb") as f:
        stages_dict = pickle.load(f)
    
    ids_list = scores['img_id']
    labels_list = scores['label']
    scores = scores['score']
    id = stages_dict['img_id']
    grades = (np.nan_to_num(stages_dict['stage'], nan =0)).astype(int)

    idx_value = {valor: np.where(id== valor)[0] for valor in ids_list}
    idx_conc = np.delete(np.sort(np.concatenate(list(idx_value.values()))), [122,123])
    grades = grades[idx_conc]

    # Segregation by grade in doctors scale: 0,1,2,3,4,5,6
    print("GRADE SEGREGATION for " + best_model_dir + "\n")
    with open(write_file, "w") as file:
        file.write("GRADE SEGREGATION for " + best_model_dir + "\n")

    grades_unique = np.unique(grades)
    ids_grades = []
    labels_grades = []
    scores_grades = []
    n_images_grades_sick = []
    n_images_grades_healthy= []
    AUCs_grades = []

    for grade in grades_unique: 
        index_grade = np.where(grades == grade)[0]
        ids_grade = ids_list[index_grade]
        scores_grade = scores[index_grade]
        labels_grade = labels_list[index_grade]

        if grade >= 2:
            AUCs_grade = roc_auc_score(labels_grade,scores_grade)
        else: 
            AUCs_grade = 0
        
        idx_sick = np.where(labels_grade==1)[0]
        idx_healthy = np.where(labels_grade==0)[0]

        ids_grades.append(ids_grade)
        scores_grades.append(scores_grade)
        labels_grades.append(labels_grade)
        AUCs_grades.append(AUCs_grade)
        n_images_grades_sick.append(len(idx_sick))
        n_images_grades_healthy.append(len(idx_healthy))
        print('GRADE ' + str(grade) + " n = " + str(len(ids_grade)) + " AUC = " + str(AUCs_grade) + "\n")
        with open(write_file, "a") as file:
            file.write('GRADE ' + str(grade) + " n = " + str(len(ids_grade)) + " AUC = " + str(AUCs_grade) + "\n")
    
    # Save results by Fuch's grade
    grade_segregation_scores = {'grade': grades_unique, 'img_id': ids_grades, 'score': scores_grades, 'label': labels_grades}
    with open(save_scores_dir + '/grade_segregation_scores.pkl', "wb") as f:
        pickle.dump(grade_segregation_scores, f)

    grade_segregation_AUCs = {'grade': grades_unique, 'AUC': AUCs_grades}
    with open(save_scores_dir + '/grade_segregation_AUCs.pkl', "wb") as f:
        pickle.dump(grade_segregation_AUCs, f)
    
    # Show number of sick and healthy patients for each grade
    width = 0.35
    fig, ax = plt.subplots()
    bar1 = ax.bar(grades_unique - width/2, n_images_grades_sick, width, label='Patológicos', color='red')
    bar2 = ax.bar(grades_unique + width/2, n_images_grades_healthy, width, label='Sanos', color='blue')

    for rect in bar1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), 
                textcoords="offset points",
                ha='center', va='bottom')

    for rect in bar2:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', va='bottom')

    ax.set_xlabel('Grados de FECD')
    ax.set_ylabel('Número de casos')
    ax.set_title('Número de casos patológicos y sanos por etapa')
    ax.set_xticks(grades_unique)
    ax.set_xticklabels(['0', '1', '2', '3', '4', '5', '6'])
    ax.legend()
    plt.savefig(save_scores_dir + '/grades_nmaps.png')

    # Segregation by stage in doctors scale: Intermediate grades=[1,2,3] and Severe grades=[4,5,6]
    print("STAGE SEGREGATION for " + best_model_dir + "\n")
    with open(write_file, "w") as file:
        file.write("STAGE SEGREGATION for " + best_model_dir + "\n")
    stages_unique = [[1,2,3], [4,5,6]]
    n_stages = np.arange(len(stages_unique))
    ids_stages = []
    labels_stages = []
    scores_stages = []
    n_images_stages_sick = []
    n_images_stages_healthy= []
    AUCs_stages = []

    for stage in stages_unique: 
        index_stage = []
        for grade in stage: 
            index_stage = np.concatenate((index_stage, np.where(grades == grade)[0]))
        index_stage = index_stage.astype(int)
        ids_stage = ids_list[index_stage]
        scores_stage = scores[index_stage]
        labels_stage = labels_list[index_stage]
        AUCs_stage = roc_auc_score(labels_stage,scores_stage)
        idx_sick = np.where(labels_stage==1)[0]
        idx_healthy = np.where(labels_stage==0)[0]
        ids_stages.append(ids_stage)
        scores_stages.append(scores_stage)
        labels_stages.append(labels_stage)
        AUCs_stages.append(AUCs_stage)
        n_images_stages_sick.append(len(idx_sick))
        n_images_stages_healthy.append(len(idx_healthy))
        print('STAGE ' + str(stage) + " n = " + str(len(ids_stage)) + " AUC = " + str(AUCs_stage) + "\n")
        with open(write_file, "a") as file:
            file.write('STAGE' + str(stage) + " n = " + str(len(ids_stage)) + " AUC = " + str(AUCs_stage) + "\n")

    # Save results by Fuch's stage
    stage_segregation_scores = {'stage': stages_unique, 'img_id': ids_stages, 'score': scores_stages, 'label': labels_stages}
    with open(save_scores_dir + '/stage_segregation_scores.pkl', "wb") as f:
        pickle.dump(stage_segregation_scores, f)

    stage_segregation_AUCs = {'stage': stages_unique, 'AUC': AUCs_stages}
    with open(save_scores_dir + '/stage_segregation_AUCs.pkl', "wb") as f:
        pickle.dump(stage_segregation_AUCs, f)
    
    # Show number of sick and healthy patients for each stage
    width = 0.35
    fig, ax = plt.subplots()
    bar1 = ax.bar(n_stages - width/2, n_images_stages_sick, width, label='Patológicos', color='red')
    bar2 = ax.bar(n_stages + width/2, n_images_stages_healthy, width, label='Sanos', color='blue')

    for rect in bar1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

    for rect in bar2:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

    ax.set_xlabel('Etapas de FECD')
    ax.set_ylabel('Número de casos')
    ax.set_title('Número de casos patológicos y sanos por etapa')
    ax.set_xticks(n_stages)
    ax.set_xticklabels(['Intermediate', 'Severe'])
    ax.legend()
    plt.savefig(save_scores_dir + '/stages_nmaps.png')

