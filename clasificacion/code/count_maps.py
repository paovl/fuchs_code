"""
@author: pvltarife
"""
import pandas as pd
from EyeDataset import EyeDataset
import pdb

if '__main__':
    # HRYC
    print("HRYC")
    fullAnnFile_hryc = '../datasets/dataset_hryc/full_annotation.csv'
    csv_File_hryc = '../datasets' + '/'+ 'dataset_hryc' + '/' + 'SEP' + '/' + 'ransac_TH_1.5_r_45' + '/annotation.csv'
    dataset = EyeDataset(csv_file=csv_File_hryc, image_dir='../datasets/dataset_hryc', imSize=(141,141))

    # csvFile_hryc = '../datasets' + '/'+ 'dataset_hryc' + '/' + 'RESNET' + '/' + 'ransac_TH_1.5_r_45' + '/annotation.csv'
    # csvFile_bio_hryc = '../datasets' + '/'+ 'dataset_hryc' + '/' + 'RESNET' + '/' + 'ransac_TH_1.5_r_45' + '/annotation_biomarkers.csv'

    columns_hryc = ["Nombre","Ojo","Ojo (OD 0 OI 1)","Descompensación corneal"]
    columns_bio_hryc = columns_hryc + ["Paqui Relativa","DensitAnt0_2"]

    df_hryc = pd.read_csv(fullAnnFile_hryc, delimiter=';', usecols=columns_hryc)
    df_hryc = df_hryc.dropna()
    print("MAPS")
    print("Number of patients: ", len(df_hryc["Nombre"].drop_duplicates()))
    print("Number of maps: ", len(df_hryc))

    df_bio_hryc = pd.read_csv(fullAnnFile_hryc, delimiter=';',usecols=columns_bio_hryc)
    df_bio_hryc = df_bio_hryc.dropna()
    print("BIOMARKERS")
    print("Number of patients: ", len(df_bio_hryc["Nombre"].drop_duplicates()))
    print("Number of maps: ", len(df_bio_hryc))

    # Multicentrico
    print("Multicentrico")
    fullAnnFile_multi = '../datasets/dataset_multicentrico/full_annotation.csv'
    # csvFile_multi = '../datasets' + '/'+ 'dataset_multicentrico' + '/' + 'RESNET' + '/' + 'ransac_TH_1.5_r_45' + '/annotation.csv'
    # csvFile_bio_multi = '../datasets' + '/'+ 'dataset_multicentrico' + '/' + 'RESNET' + '/' + 'ransac_TH_1.5_r_45' + '/annotation_biomarkers.csv'

    columns_multi = ["Archivo", "Número de Registro","Lado (OD 0 OI 1)","Hospital","Descompensación corneal","INCLUIDOS EN ESTUDIO"]
    columns_bio_multi = columns_multi + ["Paq Relativa","Densitometria ant"]

    df_multi = pd.read_csv(fullAnnFile_multi, delimiter=';', usecols=columns_multi)
    df_multi = df_multi[df_multi['INCLUIDOS EN ESTUDIO']=='0']
    df_multi = df_multi.iloc[2:]
    df_multi = df_multi.dropna()
    print("MAPS")
    print("Number of patients: ", len(df_multi["Archivo"].drop_duplicates()))
    print("Number of maps: ", len(df_multi))

    df_bio_multi = pd.read_csv(fullAnnFile_multi, delimiter=';', usecols=columns_bio_multi)
    df_bio_multi = df_bio_multi.dropna()
    df_bio_multi = df_bio_multi[df_bio_multi['INCLUIDOS EN ESTUDIO']=='0']
    df_bio_multi = df_bio_multi.iloc[2:]
    df_bio_multi = df_bio_multi.dropna()
    print("BIOMARKERS")
    print("Number of patients: ", len(df_bio_multi["Archivo"].drop_duplicates()))
    print("Number of maps: ", len(df_bio_multi))
    
    print("TOTAL")
    print("Number of patients: ", len(df_hryc["Nombre"].drop_duplicates()) + len(df_multi["Archivo"].drop_duplicates()))
    print("Number of patients with biomarkers: ", len(df_bio_hryc["Nombre"].drop_duplicates()) + len(df_bio_multi["Archivo"].drop_duplicates()))
    print("Number of maps: ", len(df_hryc) + len(df_multi))
    print("Number of maps with biomarkers: ", len(df_bio_hryc) + len(df_bio_multi))
    
