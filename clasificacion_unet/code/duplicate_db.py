# Imports
import os
import cv2
import numpy as np
import pandas as pd
import torch
import importlib
from skimage.transform import resize
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_erosion, binary_dilation, disk
from torchvision import transforms
from models_unet_duplicate import UNet
import pickle
from dataAugmentation_unet import PolarCoordinates, reSize, cropEye, ToTensor, Normalize
import matplotlib.pyplot as plt
import pdb

if __name__=='__main__':
    path = '../datasets/dataset_global/SEP/ransac_TH_1.5_r_45/annotation.csv'
    path_weights = '../../ssl_unet/code/results/unet_map_seed0_dseed0_num2_MSELoss_final/ELE_4_CORNEA-DENS_0_PAC_0/unet_self_supervised_fold.pth'
    path_model = '../../ssl_unet/code/results/unet_map_seed0_dseed0_num2_MSELoss_final/ELE_4_CORNEA-DENS_0_PAC_0/model.pth'
    path_conf = 'conf.config_70iter'
    path_normalization = '../datasets/dataset_global/SEP/ransac_TH_1.5_r_45/data/normalization.npz'

    # crear nueva carpeta
    dbs = ['../datasets/dataset_multicentrico/SEP/ransac_TH_1.5_r_45', '../datasets/dataset_hryc/SEP/ransac_TH_1.5_r_45']
    new_db = 'data_dupe'
    new_db_image = 'images_dupe'

    for i in dbs:
        new_dir = os.path.join(i, new_db)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

    for i in dbs:
        new_dir = os.path.join(i, new_db_image)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
    
    # Read the CSV file
    df = pd.read_csv(path)
    # Print the first few rows of the DataFrame
    print(df.head())

    db_paths = df['img_path'].values

    db_dupe_paths = [s.replace("/data/", "/data_dupe/") for s in db_paths]

    # Read the configuration file
    cfg = importlib.import_module(path_conf)
    max_angle = 360

    # Read model
    model = UNet(n_maps=len(cfg.mapList), n_labels=1)
    # Read weights
    weights = torch.load(path_weights)
    # Apply weights to the model
    model.load_state_dict(weights)
    model.eval()

    # Read normalization file
    with open(path_normalization, 'rb') as f:
        normalization_data = pickle.load(f)

    idxCat = [normalization_data['categories'].index(cat) for cat in cfg.mapList]
    nMaps = len(cfg.mapList)

    normalizer = Normalize(mean=normalization_data['mean'][idxCat],std=normalization_data['std'][idxCat])

    # Transformations 
    list_transforms=[reSize(cfg.imSize),
                         PolarCoordinates(cfg.imSize[0]/2 , float(max_angle)),
                         ToTensor(),
                         normalizer]
   
    # Crop image border
    list_transforms = [cropEye(cfg.mapList,cfg.cropEyeBorder)] + list_transforms 

    # Combine all transformations in a basic chain
    transform_chain = transforms.Compose(list_transforms)

    for i in range(len(df)):
        # Read the image
        img_path = db_paths[i]
        img_dupe_path = db_dupe_paths[i]
        img_id = df.iloc[i]['img_id']
        label = df.iloc[i]['label']
        
        # Read the image
        image = np.empty((141, 141, nMaps), dtype=np.float32)
        for j in range(nMaps):
            impath='%s_%s.npy'%(img_path, cfg.mapList[j])
            cmap=np.load(impath)
            (h,w)=cmap.shape
            if w!=141 or h!=141:
                cmap=resize(cmap,(141, 141),preserve_range=True)  
            image[:,:,j] = cmap
        
        sample = {'id': img_id,'image': image, 'label':float(label), 'img_path': img_path}

        sample = transform_chain(sample)

        # pdb.set_trace()

        image = sample['image']

        # mask_bg = sample['mask_bg']

        # mask_in = ~mask_bg

        # mask_dilation = mask_in.clone().detach().numpy()

        # # Calculate dilation
        # for i in range(nMaps):
        #     mask_dilation[i, :, :] = binary_dilation(mask_dilation[i, :, :], footprint=disk(3))
        #     mask_dilation[i, :, :] = binary_fill_holes(mask_dilation[i, :, :])
        
        # mask_dilation = mask_dilation.astype(bool)

        # mask_dilation_bg = ~mask_dilation

        masked_image = image.clone()

        augmented_image = image.clone()
       
        for c in range(nMaps):
            # select the oder two maps and enmascarate this
            new_image = image.clone()
            new_image[c, :, :] = 0

            # normalize again
            mean = normalizer.mean[c] 
            std = normalizer.std[c]

            new_image[c, :, :] = (new_image[c, :, :] - mean) / std
            masked_image[c, :, :] = new_image[c, :, :]
            
            with torch.no_grad():
                output_image = model(new_image.unsqueeze(0)).squeeze()
            
            augmented_image[c, :, :] = output_image[c, :, :]

            # Reestablecemos el fondo
            # augmented_image[c, :, :][mask_dilation_bg[c, :, :]] = 0
            # augmented_image[c, :, :][mask_dilation_bg[c, :, :]] = (augmented_image[c, :, :][mask_dilation_bg[c, :, :]] -mean) / std

            # save each map of the augmented image
            augmented_img_file = '%s_%s.npy'%(img_dupe_path, cfg.mapList[c])
            np.save(augmented_img_file, augmented_image[c, :, :].detach().numpy())

            # save image

            augmented_image_plot = augmented_image[c, :, :].clone()
            augmented_image_plot = augmented_image_plot *std + mean
            augmented_image_plot = torch.clamp(augmented_image_plot, 0, 1)

            augmented_image_plot_path = '%s_%s.png'%(img_dupe_path, cfg.mapList[c])
            augmented_image_plot_path  = augmented_image_plot_path.replace("/data_dupe/", "/images_dupe/")

            cv2.imwrite(augmented_image_plot_path, np.uint16(65535*augmented_image_plot.detach().numpy()))

        augmented_image = normalizer.denormalize(augmented_image)
        augmented_image = torch.clamp(augmented_image, 0, 1)
        masked_image = normalizer.denormalize(masked_image)
        masked_image = torch.clamp(masked_image, 0, 1)
        image = normalizer.denormalize(image)
        image = torch.clamp(image, 0, 1)

        augmented_image = augmented_image.detach().cpu().numpy()
        image = image.detach().cpu().numpy()

        # plot every map 
        fig, axes = plt.subplots(2, nMaps, figsize=(15, 5))
        for j in range(nMaps):
            axes[0, j].imshow(image[j, :, :], cmap='gray', vmin=0, vmax=1)
            axes[0, j].set_title(f'Original Map {j+1}')
            axes[0, j].axis('off')

            axes[1, j].imshow(augmented_image[j, :, :], cmap='gray', vmin=0, vmax=1)
            axes[1, j].set_title(f'Augmented Map {j+1}')
            axes[1, j].axis('off')
        plt.tight_layout()
        plt.savefig(img_dupe_path.replace('data_dupe', 'images_dupe') + ".png", dpi=300)
        plt.close(fig)

    df_dupe = df.copy()
    df_dupe['img_path'] = db_dupe_paths

    output_csv_path = os.path.join(os.path.dirname(path), 'annotation_dupe.csv')
    df_dupe.to_csv(output_csv_path, index=False)





        









