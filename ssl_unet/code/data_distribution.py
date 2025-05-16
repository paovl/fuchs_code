from EyeDataset_unet import EyeDataset
import os
import torch
import numpy as np
from torchvision import transforms
import pickle
from dataAugmentation_unet import PolarCoordinates, reSize, cropEye, ToTensor, Normalize
import pdb

if __name__ == '__main__':
    # Ver la fdp de los distintos mapas tomográficos de toda la base de datos
    # Cargar la base de datos
    csvFile = '../datasets/dataset_global_unet/patient_list.csv'
    db_path = '../datasets'
    dataDir = 'data'
    mapList = ['ELE_4', 'CORNEA-DENS_0', 'PAC_0']
    imSize = (224, 224)
    
    normalization_file = os.path.join(db_path,'dataset_global_unet/data/normalization.npz')
    with open(normalization_file, 'rb') as f:
        normalization_data = pickle.load(f)
    #Filter using the map list
    idxCat = [normalization_data['categories'].index(cat) for cat in mapList]
    mean = normalization_data['mean'][idxCat]
    std = normalization_data['std'][idxCat]

    print('Map list: ', mapList)
    print('Mean: ', mean)
    print('Std: ', std)

    # transform_chain = transforms.Compose([ cropEye(mapList, border=10),
    #                                         reSize(imSize),
    #                                         PolarCoordinates(),
    #                                         ToTensor(),
    #                                         Normalize(mean=mean, std=std)])

    transform_chain = transforms.Compose([cropEye(mapList, border=10),
                                            reSize(imSize),
                                            PolarCoordinates(), ToTensor(), Normalize(mean=mean, std=std)])
    full_dataset = EyeDataset(csvFile, db_path, dataDir, transform = transform_chain, mapList = mapList)

    # Calcular la fdp de los mapas tomográficos de la base de datos
    ele4_maps = []
    cornea_dens_0_maps = []
    pac_0_maps = []

    for i in range(len(full_dataset)):
        sample = full_dataset[i]['img']
        ele4_maps.append(sample[0, :, :])
        cornea_dens_0_maps.append(sample[1, :, :])
        pac_0_maps.append(sample[2, :, :])
    
    # Calculate fdp and plot

    import matplotlib.pyplot as plt
    import seaborn as sns

    ele4_maps = torch.cat(ele4_maps)
    # ele4_mask = ele4_maps<-800
    # ele4_maps[ele4_mask].max()
    # ele4_maps[ele4_mask].min()
    # ele4_maps[~ele4_mask].min()
    # ele4_maps[~ele4_mask].max()

    # print('ELE_4')
    # print('Max in: ', ele4_maps[~ele4_mask].max())
    # print('Min in: ', ele4_maps[~ele4_mask].min())
    # print('Max out: ', ele4_maps[ele4_mask].max())
    # print('Min out: ', ele4_maps[ele4_mask].min())
    # print("Num pixels background: ", torch.sum(ele4_mask))

    cornea_dens_0_maps = torch.cat(cornea_dens_0_maps)
    # cornea_dens_0_mask = cornea_dens_0_maps<-800
    # cornea_dens_0_maps[cornea_dens_0_mask].max()
    # cornea_dens_0_maps[cornea_dens_0_mask].min()
    # cornea_dens_0_maps[~cornea_dens_0_mask].min()
    # cornea_dens_0_maps[~cornea_dens_0_mask].max()

    # print('CORNEA-DENS_0')
    # print('Max in: ', cornea_dens_0_maps[~cornea_dens_0_mask].max())
    # print('Min in: ', cornea_dens_0_maps[~cornea_dens_0_mask].min())
    # print('Max out: ', cornea_dens_0_maps[cornea_dens_0_mask].max())
    # print('Min out: ', cornea_dens_0_maps[cornea_dens_0_mask].min())
    # print("Num pixels background: ", torch.sum(cornea_dens_0_mask))

    pac_0_maps = torch.cat(pac_0_maps)
    # pac_0_mask = pac_0_maps<-800
    # pac_0_maps[pac_0_mask].max()
    # pac_0_maps[pac_0_mask].min()
    # pac_0_maps[~pac_0_mask].min()
    # pac_0_maps[~pac_0_mask].max()

    # print('PAC_0')
    # print('Max in: ', pac_0_maps[~pac_0_mask].max())
    # print('Min in: ', pac_0_maps[~pac_0_mask].min())
    # print('Max out: ', pac_0_maps[pac_0_mask].max())
    # print('Min out: ', pac_0_maps[pac_0_mask].min())
    # print("Num pixels background: ", torch.sum(pac_0_mask))

    map_names = ['ELE_4', 'CORNEA-DENS_0', 'PAC_0']
    map_data = [ele4_maps, cornea_dens_0_maps, pac_0_maps]

    # plt.figure(figsize =(10, 6))

    # for data, name in zip(map_data, map_names):
    #     data = data.numpy().flatten()  # Convertir a numpy y a 1D
    #     sns.kdeplot(data, label=name, fill=True, alpha=0.4)  # FDP con suavizado

    # plt.title("Función de Densidad de Probabilidad de los Mapas Tomográficos")
    # plt.xlabel("Valor del píxel")
    # plt.ylabel("Densidad")
    # plt.xlim(-20, 20)
    # plt.legend()
    # plt.grid()
    # plt.show()

   # Calcular el histograma con resolución de 0.1
    bins = np.arange(-20, 20, 0.1)  # Definir los bins de -17 a 17 con paso 0.1
    histograma_ele4, _ = np.histogram(ele4_maps.ravel(), bins=bins)  # Calcular el histograma
    fdp_ele4 = histograma_ele4 / np.sum(histograma_ele4)  # Normalizar el histograma
    histograma_cornea_dens_0, _ = np.histogram(cornea_dens_0_maps.ravel(), bins=bins)  # Calcular el histograma
    fdp_cornea_dens_0 = histograma_cornea_dens_0 / np.sum(histograma_cornea_dens_0)  # Normalizar el histograma
    histograma_pac_0, _ = np.histogram(pac_0_maps.ravel(), bins=bins)  # Calcular el histograma
    fdp_pac_0 = histograma_pac_0 / np.sum(histograma_pac_0)  # Normalizar el histogram

    # Graficar el histograma
    # plt.figure(figsize=(8,5))
    # plt.bar(bins[:-1], histograma_ele4, width=0.1, label='ELE_4', color='blue', alpha=0.7) # width=0.1 para coincidir con los bins
    # plt.bar(bins[:-1], histograma_cornea_dens_0, width=0.1, label='CORNEA-DENS_0', color='red', alpha=0.7)
    # plt.bar(bins[:-1], histograma_pac_0, width=0.1, label='PAC_0', color='green', alpha=0.7)
    # plt.xlabel("Valor de Intensidad de Píxeles")
    # plt.ylabel("Frecuencia (Número de Píxeles)")
    # plt.title("Histograma de Intensidades de Píxeles en Escala Original")
    # plt.legend()
    # plt.grid(axis='y')
    # plt.show()

    # Graficar la fdp   
    plt.figure(figsize=(8,5))
    plt.bar(bins[:-1], fdp_ele4, label='ELE_4',  width=0.1, color='blue', alpha=0.7)
    plt.bar(bins[:-1], fdp_cornea_dens_0,  width=0.1, label='CORNEA-DENS_0', color='red', alpha=0.7)
    plt.bar(bins[:-1], fdp_pac_0, label='PAC_0',  width=0.1, color='green', alpha=0.7)
    plt.xlabel("Valor de Intensidad de Píxeles")
    plt.ylabel("Densidad de Probabilidad")
    plt.title("Función de Densidad de Probabilidad de los Mapas Tomográficos")
    plt.legend()
    plt.grid(axis='y')
    plt.show()

    
    
