"""
@author: pvltarife
"""
import pandas as pd
from PairEyeDataset import PairEyeDataset
import pdb

if '__main__':
    # HRYC
    print("SSL")
    fullAnnFile_ssl = '../datasets/dataset_ssl/patient_list.csv'
    # csvFile_ssl = '../datasets' + '/'+ 'dataset_ssl' + '/' + 'RESNET' + '/' + 'ransac_TH_1.5_r_45' + '/annotation.csv'
    dataset = PairEyeDataset(fullAnnFile_ssl, '../datasets/dataset_ssl', 'image', 'data', 'error_PAC', 'error_BFS')

    columns_ssl = ["Nombre","Patient","Ojo","Sesion","Temp","Database"]

    df_ssl = pd.read_csv(fullAnnFile_ssl, delimiter=',', usecols=columns_ssl)
    df_ssl = df_ssl.dropna()
    # # Delete those that hava no multiple sessions
    patients_pair = df_ssl[df_ssl['Temp'] == 2]['Patient'].unique()
    df_ssl = df_ssl[df_ssl['Patient'].isin(patients_pair)]
    print("MAPS")
    print("Number of patients: ", len(df_ssl["Patient"].drop_duplicates()))
    print("Number of maps: ", len(df_ssl))
    print("Number of right records: ", len(df_ssl[df_ssl["Ojo"] == "OD"]))
    print("Number of left records: ", len(df_ssl[df_ssl["Ojo"] == "OS"]))

    # Multicentrico
    print("Multicentrico")
    fullAnnFile_multi = '../datasets/dataset_multicentrico/patient_list.csv'
    # csvFile_multi = '../datasets' + '/'+ 'dataset_multicentrico' + '/' + 'RESNET' + '/' + 'ransac_TH_1.5_r_45' + '/annotation.csv'

    columns_multi = ["Nombre","Patient","Ojo","Sesion","Temp","Database"]
    df_multi = pd.read_csv(fullAnnFile_multi, delimiter=',', usecols=columns_multi)
    df_multi = df_multi.dropna()
    pdb.set_trace()
    print("MAPS")
    print("Number of patients: ", len(df_multi["Patient"].drop_duplicates()))
    print("Number of maps: ", len(df_multi))
    print("Number of right records: ", len(df_multi[df_multi["Ojo"] == "OD"]))
    print("Number of left records: ", len(df_multi[df_multi["Ojo"] == "OS"]))

    # TOTAL 
    print("TOTAL")
    print("Number of patients: ", len(df_ssl["Patient"].drop_duplicates()) + len(df_multi["Patient"].drop_duplicates()))
    print("Number of maps: ", len(df_ssl) + len(df_multi))
    print("Number of right records: ", len(df_ssl[df_ssl["Ojo"] == "OD"]) + len(df_multi[df_multi["Ojo"] == "OD"]))
    print("Number of left records: ", len(df_ssl[df_ssl["Ojo"] == "OS"]) + len(df_multi[df_multi["Ojo"] == "OS"]))




