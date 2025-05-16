"""
Created on Fri Feb 09 2024
@author: pvltarife
"""
# Imports
import sys
import re
import os
import pandas as pd
import glob 
import pdb
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_erosion
import pickle
import itertools
import math
from skimage.morphology import binary_dilation, disk
from skimage.measure import label as label_mask
from skimage.measure import regionprops
from skimage.filters import gaussian
from scipy.optimize import minimize
from scipy.linalg import lstsq
from itertools import zip_longest
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)

# Global variables

# Max and min values for each map category
PAC_max = 0
PAC_min = float('inf')

CUR_max = np.zeros((5,))
CUR_min = np.full((5,), float('inf'))

ELE_max = np.zeros((6,))
ELE_min = np.full((6,), float('inf'))

ELE_BFS_max = np.zeros((6,))
ELE_BFS_min = np.full((6,), float('inf'))

CORNEA_DENS_max = np.zeros((2,))
CORNEA_DENS_min = np.full((2,), float('inf'))

ELE_BFS_8mm_max = np.zeros((2,))
ELE_BFS_8mm_min = np.full((2,), float('inf'))

def find_folder(base_path, folder_name):
    folders = os.listdir(base_path)
    for folder in folders:
        if folder.lower() == folder_name.lower():
            return os.path.join(base_path, folder)
    return None
def readCSV(csv_path, csv):

    #Read the csv file
    df = pd.read_csv(os.path.join(csv_path,csv), delimiter=';',
                     decimal=',',encoding='unicode_escape')
    
    # Assign values for background pixels 
    if csv.find('ELE_BFS_8mm') > 0: # ELE maps from Pentacam have negative values
        df = df.fillna(-200)
    else:
        df = df.fillna(0)

    end = df[df.iloc[:, 0] == '[SYSTEM]'].index.tolist()[0] 
    dataset = df.iloc[:end]
    dataset = dataset.replace(',', '.', regex=True)
    
    return dataset

def computeMaxMin(dataset, fname, outname):
    """
    This functions goes through all maps in order to get max and min for each map category
    """
    global PAC_max, PAC_min, CUR_max, CUR_min, ELE_max, ELE_min, CORNEA_DENS_max, CORNEA_DENS_min, ELE_BFS_8mm_max, ELE_BFS_8mm_min

    # Aux variables 
    i = 0
    j = 0

    step = 141 # Image rows
    iterations = int(dataset.shape[0]/step) # Count rows in the file
    
    od = fname.find('_OD_') > 0 # OD (right eye)

    for k in range(iterations):

        j = i + step 
        array = dataset[i:j].to_numpy()
        pixels = array[:,1:]
        i = j + 1 
        pixels = pixels.astype(np.float64)

        if od:  
            pixels = pixels[:-1:] # Flip map 

        if fname.find('ELE_BFS_8mm') > 0:
            mask = pixels > -200  # Gets a mask for pixels with greater values than -200
            mask = binary_fill_holes(mask) 
        else:
            mask = pixels > 0  # Gets a mask for pixels with greater values than 0
            mask = binary_fill_holes(mask) 

        max_value = pixels[mask].max()
        min_value = pixels[mask].min()

        if fname.find('ELE_BFS_8mm') > 0:

            if(max_value > ELE_BFS_8mm_max[k]):
                ELE_BFS_8mm_max[k] = max_value

            if(min_value < ELE_BFS_8mm_min[k]):
                ELE_BFS_8mm_min[k] = min_value

        elif fname.find('ELE') > 0:

            if(max_value > ELE_max[k]):
                ELE_max[k] = max_value

            if(min_value < ELE_min[k]):
                ELE_min[k] = min_value

        elif fname.find('PAC') > 0:

            if(max_value > PAC_max):
                PAC_max = max_value

            if(min_value < PAC_min):
                PAC_min = min_value

        elif fname.find('CUR') > 0:

            if(max_value > CUR_max[k]):
                CUR_max[k] = max_value

            if(min_value < CUR_min[k]):
                CUR_min[k] = min_value

        elif fname.find('CORNEA-DENS') > 0:

            if(max_value > CORNEA_DENS_max[k]):
                CORNEA_DENS_max[k] = max_value

            if(min_value < CORNEA_DENS_min[k]):
                CORNEA_DENS_min[k] = min_value

def computeMaxMinBFS(fpath, dataset, fname, outname, info_dataset, rng, th = 1.5, diskRadius= 45, ransac = 1, px = 100): 
    """
    This functions goes through all ELE_3 and ELE_4 maps in order to get max and min values for the BFS maps
    """

    global PAC_max, PAC_min, ELE_max, ELE_min, ELE_BFS_max, ELE_BFS_min

    # Aux variables 
    i = 0
    j = 0

    step = 141 # Image rows
    iterations = int(dataset.shape[0]/step) # Count rows in the file

    fname_split = fname.split('_')

    od = fname.find('_OD_') > 0 # OD (right eye)
    ele = fname[-3:] == 'ELE' # ELE MAP
    pac = fname[-3:] == 'PAC' # PAC MAP

    # Get label 
    if fname_split[0] in info_dataset['img_id'].values:
        label = info_dataset['label'][info_dataset['img_id'] == fname_split[0]].values[0]
    else:
        label = -1 

    for k in range(iterations):

        j = i + step 
        array = dataset[i:j].to_numpy() 
        pixels = array[:,1:] 
        i = j + 1 
        pixels = pixels.astype(np.float64)

        if od:  
            pixels = pixels[:-1:] # Flip map
        
        mask = pixels > 0  # Gets a mask for pixels with greater values than 0
        mask = binary_fill_holes(mask)

        if pac:
            pixels = (pixels - PAC_min) / (PAC_max - PAC_min)
            # Save data
            pixels[~mask] = - px
            np.save(os.path.join(fpath,outname + '_' + str(k) + '.npy'), pixels)

        if ele and ransac != -1 and (k == 4 or k == 3):

            # RANSAC algorithm with sphere fit
            if ransac == 1:
                pixels, _ = ransac_bfs(pixels, rng, label, th = th, diskRadius= diskRadius, state = 1)

            # No RANSAC with quadratic fit
            if ransac == 0: 
                pacimage = np.load(os.path.join(fpath,outname.replace('ELE','PAC') + '_0.npy')) # Recover PAC map
                pixels_rel = quadraticFit(pixels, pacimage, diskRadius= diskRadius) 
                pixels = pixels_rel

            max_value = pixels[mask].max()
            min_value = pixels[mask].min()

            if(max_value > ELE_BFS_max[k]):
                ELE_BFS_max[k] = max_value

            if(min_value < ELE_BFS_min[k]):
                ELE_BFS_min[k] = min_value
 
def dataGenerator(fpath, error_BFS_path, error_PAC_path, ipath, hpath, ppath, dataset, fname, outname, info_dataset, rng,  th = 1.5, diskRadius= 45, ransac = 1, px = 100):

    global PAC_max, PAC_min, CUR_max, CUR_min, ELE_max, ELE_min, CORNEA_DENS_max, CORNEA_DENS_min, ELE_BFS_max, ELE_BFS_min,ELE_BFS_8mm_min, ELE_BFS_8mm_max
    
    # Aux variables 
    i = 0
    j = 0

    step = 141 # Image rows
    iterations = int(dataset.shape[0]/step) # Count rows in file

    fname_split = fname.split('_')

    od = fname.find('_OD_') > 0 # OD (right eye)
    ele = fname[-3:] == 'ELE' # ELE map
    pac = fname.find('_PAC')>0

    # Get label
    if fname_split[0] in info_dataset['img_id'].values:
        label = info_dataset['label'][info_dataset['img_id'] == fname_split[0]].values[0]
    else:
        label = -1 

    for k in range(iterations):

        j = i + step 
        array = dataset[i:j].to_numpy() 
        pixels = array[:,1:] 
        i = j + 1 
        pixels = pixels.astype(np.float64)

        if od:  
            pixels = pixels[:-1:] # Flip map
        
        if fname.find('ELE_BFS_8mm') > 0:
            mask = pixels > -200  # Gets a mask for pixels with greater values than -200
            mask = binary_fill_holes(mask) 
        else:
            mask = pixels > 0  # Gets a mask for pixels with greater values than 0
            mask = binary_fill_holes(mask) 

        # Normalization 

        if fname.find('ELE_BFS_8mm') > 0:
            pixels = (pixels - ELE_BFS_8mm_min[k]) / (ELE_BFS_8mm_max[k] - ELE_BFS_8mm_min[k])

        elif fname.find('ELE') > 0 and (ransac == -1 or (k != 4 and k != 3)):
            pixels = (pixels - ELE_min[k]) / (ELE_max[k] - ELE_min[k])
            
        elif fname.find('PAC') > 0:
            pixels = (pixels - PAC_min) / (PAC_max - PAC_min)

        elif fname.find('CUR') > 0:
            pixels = (pixels - CUR_min[k]) / (CUR_max[k] - CUR_min[k])

        elif fname.find('CORNEA-DENS') > 0:
            pixels = (pixels - CORNEA_DENS_min[k]) / (CORNEA_DENS_max[k] - CORNEA_DENS_min[k])
        
        if pac:
            _, error_PAC = ransac_pac(pixels, rng, th = th, state = 1)
            if od:
                np.save(os.path.join(error_PAC_path,outname + '_' + str(k) + '.npy'), error_PAC)
            else:
                np.save(os.path.join(error_PAC_path,outname + '_' + str(k) + '.npy'), error_PAC)
        
        if ele and ransac != -1 and (k == 4 or k == 3):

            # RANSAC algorithm with sphere fit
            if ransac == 1:
                pixels, error_BFS= ransac_bfs(pixels, rng, label, th = th, diskRadius= diskRadius)
                if od:
                    np.save(os.path.join(error_BFS_path,outname + '_' + str(k) + '.npy'), error_BFS)
                else:
                    np.save(os.path.join(error_BFS_path,outname + '_' + str(k) + '.npy'), error_BFS)
            # No RANSAC algorithm with quadratic fit
            if ransac == 0: 
                pacimage = np.load(os.path.join(fpath,outname.replace('ELE','PAC') + '_0.npy')) # Recover PAC map
                pixels_rel = quadraticFit(pixels, pacimage, diskRadius= diskRadius) 
                pixels = pixels_rel
            
            pixels = (pixels - ELE_BFS_min[k]) / (ELE_BFS_max[k] - ELE_BFS_min[k]) # Normalization
            pixels = 1 - pixels

        pixels_img = pixels.copy()

        # Save data
        pixels[~mask] = - px
        np.save(os.path.join(fpath,outname + '_' + str(k) + '.npy'), pixels)
        
        pixels_img[~mask] = 0

        if outname.find('CORNEA-DENS') > 0 and k == 0:
            pixels_img[~mask] = (0 - CORNEA_DENS_min[0]) / (CORNEA_DENS_max[0] - CORNEA_DENS_min[0])
        elif outname.find('PAC') > 0 and k == 0:
            pixels_img[~mask] = (452.0 - PAC_min) / (PAC_max - PAC_min)
        elif outname.find('ELE') > 0 and k == 4:
            if ransac == -1: 
                pixels_img[~mask] = (0.0 - ELE_min[4]) / (ELE_max[4] - ELE_min[4])
            elif ransac == 1:
                pixels_img[~mask] = 1 - ((0.335643123803089 - ELE_BFS_min[4]) / (ELE_BFS_max[4] - ELE_BFS_min[4]))
        
        imageGenerator(ipath, pixels_img, outname + '_' + str(k) + '.png', label, ransac = ransac)
        polarMapGenerator(ipath, ppath, outname + '_' + str(k) + '.png', label)

        if fname.find('PAC') > 0 and k == 0:
            pixels_img[~mask] = (1237.0 - PAC_min) / (PAC_max - PAC_min)
            heatMapGenerator(hpath, pixels_img, outname + '_' + str(k) + '.png', label)
        elif fname.find('ELE') > 0 and k == 4 and ransac == -1:
            pixels_img[~mask] = (3.386314919 - ELE_min[4]) / (ELE_max[4] - ELE_min[4])

        heatMapGenerator(hpath, pixels_img, outname + '_' + str(k) + '.png', label, ransac = ransac)

def imageGenerator(ipath, pixels, outname, label, ransac = 1):
    global PAC_max, PAC_min, CUR_max, CUR_min, ELE_max, ELE_min, CORNEA_DENS_max, CORNEA_DENS_min,  ELE_BFS_max, ELE_BFS_min
    
    if outname.find('CORNEA-DENS') > 0 and outname.find('_0.png') > 0:
        vmin = (0 - CORNEA_DENS_min[0]) / (CORNEA_DENS_max[0] - CORNEA_DENS_min[0])
        vmax = (100 - CORNEA_DENS_min[0]) / (CORNEA_DENS_max[0] - CORNEA_DENS_min[0])
    elif outname.find('PAC') > 0 and outname.find('_0.png') > 0:
        vmin = (452.0 - PAC_min) / (PAC_max - PAC_min)
        vmax = (1237.0 - PAC_min) / (PAC_max - PAC_min)
    elif outname.find('ELE') > 0 and outname.find('_4.png') > 0:
        if ransac == -1:
            vmin = (0.0 - ELE_min[4]) / (ELE_max[4] - ELE_min[4])
            vmax = (3.386314919 - ELE_min[4]) / (ELE_max[4] - ELE_min[4])
        else:
            vmin = 1 - ((0.335643123803089 - ELE_BFS_min[4]) / (ELE_BFS_max[4] - ELE_BFS_min[4]))
            vmax = 1 - ((-0.43819866569265153 - ELE_BFS_min[4]) / (ELE_BFS_max[4] - ELE_BFS_min[4]))
            
    else:
        vmin = 0
        vmax = 1

    # Directory
    if label == 1:
            ipath = ipath + '/sick'
    elif label == 0:
        ipath = ipath + '/healthy'
    else:
        ipath = ipath + '/noLabel'

    # cv2.imwrite(os.path.join(ipath,outname), np.uint16(65535*pixels), vmin=vmin, vmax=vmax)

    img.imsave(os.path.join(ipath, outname), pixels.astype(np.float64),cmap = 'gray', vmin=vmin, vmax=vmax)

def heatMapGenerator(hpath, pixels, outname, label, ransac = 1):
    global PAC_max, PAC_min, CUR_max, CUR_min, ELE_max, ELE_min, CORNEA_DENS_max, CORNEA_DENS_min,  ELE_BFS_max, ELE_BFS_min
    
    if outname.find('CORNEA-DENS') > 0 and outname.find('_0.png') > 0:
        vmin = (0 - CORNEA_DENS_min[0]) / (CORNEA_DENS_max[0] - CORNEA_DENS_min[0])
        vmax = (100 - CORNEA_DENS_min[0]) / (CORNEA_DENS_max[0] - CORNEA_DENS_min[0])
    elif outname.find('PAC') > 0 and outname.find('_0.png') > 0:
        vmin = (452.0 - PAC_min) / (PAC_max - PAC_min)
        vmax = (1237.0 - PAC_min) / (PAC_max - PAC_min)
    elif outname.find('ELE') > 0 and outname.find('_4.png') > 0:
        if ransac == -1:
            vmin = (0.0 - ELE_min[4]) / (ELE_max[4] - ELE_min[4])
            vmax = (3.386314919 - ELE_min[4]) / (ELE_max[4] - ELE_min[4])
        else:
            vmin = 1 - ((0.335643123803089 - ELE_BFS_min[4]) / (ELE_BFS_max[4] - ELE_BFS_min[4]))
            vmax = 1 - ((-0.43819866569265153 - ELE_BFS_min[4]) / (ELE_BFS_max[4] - ELE_BFS_min[4]))
    else:
        vmin = 0
        vmax = 1

    # Directory 
    if label == 1:
        hpath = hpath + '/sick'
    elif label == 0:
        hpath = hpath + '/healthy'
    else:
        hpath = hpath + '/noLabel'

    if (outname.find('PAC') > 0 and outname.find('_0.png') > 0) or (outname.find('ELE') > 0 and outname.find('_4.png') > 0 and ransac == -1):
        img.imsave(os.path.join(hpath, outname), pixels.astype(np.float64),cmap = 'jet_r', vmin=vmin, vmax=vmax)
    else:
        img.imsave(os.path.join(hpath, outname), pixels.astype(np.float64),cmap = 'jet', vmin=vmin, vmax=vmax)

def polarMapGenerator(ipath, ppath, outname, label):

    # Directory
    if label == 1:
        ppath = ppath + '/sick'
        ipath = ipath + '/sick'
    elif label == 0:
        ppath = ppath + '/healthy'
        ipath = ipath + '/healthy'
    else:
        ppath =ppath + '/noLabel'
        ipath =ipath + '/noLabel'

    image = cv2.imread(os.path.join(ipath,outname))
    image  = image.astype(np.float32)/255.0
    value = np.sqrt(((image.shape[0]/2.0)**2.0)+((image.shape[1]/2.0)**2.0))
    polar_image = cv2.linearPolar(image,(image.shape[0]/2, image.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)
    cv2.imwrite(os.path.join(ppath,outname), np.uint16(65535*polar_image))

def ransac_bfs(eleMap, rng, img_label, k=10, th = 0.05, diskRadius= 0, state = 0):
    """
    Params: 
        eleMap - elevation map 
        rng - random number generator for random data partition
        s - the minimum number of image points required to fit the model
        k - the maximum number of iterations allowed in the algorithm
        th - a threshold value to determinate when a data point fits in the model
    Return: 
        dif_image - difference between the best fit sphere and the examined cornea
    """
    # Global variables
    global acc_fit_error_healthy, acc_fit_error_sick, n_data_healthy, n_data_sick


    # RANSAC variables
    iterations = 0 
    s = 4
    max_inliers = 0 
    p = 0.99 # Probability of having a set free of outliers
    e = 0 # Outliers ratio


    # Generate mask for ele map
    mask = eleMap > 0 # Boolean map
    mask = binary_fill_holes(mask)
    mask_ones = mask.astype(int)

    #plt.figure(1);plt.imshow(eleMap); plt.title('ELE map'); plt.show() # plot ELE map

    # Disk used to estimate BFS map
    if diskRadius> 0:

       # Get the center of the map
        labeled_mask = label_mask(mask_ones)
        centroid = np.round(regionprops(labeled_mask)[0].centroid).astype(int)
        mask_est = np.zeros_like(mask)
        mask_est[centroid[0], centroid[1]] = 1
        mask_est = binary_dilation(mask_est,footprint = disk(diskRadius))
        mask_est = np.logical_and(mask,mask_est)
    else:
        mask_est = mask   
    #plt.figure(2);plt.imshow(mask_est); plt.title('MASK'); plt.show()

    # Get data coordinates
    px_coords = np.where(mask) 
    z = eleMap[mask] 
    x = px_coords[1]
    y = px_coords[0] 

    # Tranform into mm
    x_mm = x * 14 /141
    y_mm = y * 14 /141

    # Get data coordinates for disk area 
    px_coords_est = np.where(mask_est) 
    z_est = eleMap[mask_est]
    x_est = px_coords_est[1]
    y_est = px_coords_est[0]

    # Tranform into mm
    x_est_mm = x_est * 14 /141
    y_est_mm = y_est * 14 /141

    # RANSAC variables 
    s_data = len(z_est)
    k = float('inf')

    # RANSAC loop
    while iterations < k: 

        idxs = random_partition(s, s_data, rng)

        # Get the maybe inliers
        x_s = x_est[idxs] 
        y_s = y_est[idxs]
        z_s = z_est[idxs] 

        # Tranform into mm
        x_s_mm = x_s * 14 /141
        y_s_mm = y_s * 14 /141

        # Sphere fit G x m = f
        G_s = poly_matrix_RANSAC(x_s_mm, y_s_mm, z_s, s)
        f_s = np.zeros((len(x_s),1))
        f_s[:,0] = (x_s_mm**2) + (y_s_mm**2) + (z_s**2) #   Assemble the f vector
        m_s = np.linalg.lstsq(G_s, f_s ,rcond=None)[0] # Fit sphere with s points (maybe inliers)

        R_s = math.sqrt(m_s[0][0]**2 + m_s[1][0]**2 + m_s[2][0]**2 + m_s[3][0]) # Sphere radius
        xc_s_mm = m_s[0][0] # Center x point
        yc_s_mm = m_s[1][0] # Center y point
        zc_s = m_s[2][0] # Center z point

        distances_s = np.sqrt((y_mm - yc_s_mm)**2 + (x_mm - xc_s_mm)**2)
        max_distance_s = np.max(distances_s)

        # Get bfs map (maybe inliers)
        if R_s >= max_distance_s:
            z_bfs_maybe = zc_s - np.sqrt(R_s**2 - (x_mm -xc_s_mm)**2 - (y_mm -yc_s_mm)**2)
        else:
            continue

        bfs_maybe = np.zeros_like(eleMap)
        bfs_maybe[mask] = z_bfs_maybe

        #plt.figure(3);plt.imshow(bfs_maybe); plt.title('BFS maybe inliers'); plt.show()

        # Get only estimation points
        z_bfs_maybe_est = bfs_maybe[mask_est]

        # Calculate error 
        error = np.abs(z_est - z_bfs_maybe_est)
        also_inliers = np.where(error < th)[0]

        # Evaluate which of them are also inliers
        if len(also_inliers) > max_inliers:

            # Save model with max number of inliers
            max_inliers = len(also_inliers) 
            bfs_best_maybe = bfs_maybe
            idxs_maybeinliers = idxs

            # Get also inliers data points
            idxs_alsoinliers = also_inliers
            x_in_mm = x_est_mm[idxs_alsoinliers]
            y_in_mm = y_est_mm[idxs_alsoinliers]
            z_in = z_est[idxs_alsoinliers]

            # Update number of iterations (k)
            e = (s_data - max_inliers) / s_data # Number of outliers
            if (1 - (1 - e) ** s) != 0:
                if( math.log(1-(1-e)**s) != 0):
                    k = (math.log(1 - p)) / (math.log(1 - (1 - e)**s)) 
                else: 
                    k = float("inf")
            else:
                k = 0
            
        iterations += 1

    #plt.figure(4);plt.imshow(bfs_best_maybe); plt.title('BFS maybe inliers'); plt.show()

    # Estimate bfs map with the inliers of the best model 
    # Sphere fit G x m = f
    G_in = poly_matrix_RANSAC(x_in_mm, y_in_mm, z_in, s) # Asssemble G matrix 
    f_in = np.zeros((len(x_in_mm),1)) 
    f_in[:,0] = (x_in_mm**2) + (y_in_mm**2) + (z_in**2) # Asssemble f vector
    m_bfs = np.linalg.lstsq(G_in, f_in ,rcond=None)[0] # Fit sphere (also inliers from best model)
    
    R_bfs = math.sqrt(m_bfs[0][0]**2 + m_bfs[1][0]**2 + m_bfs[2][0]**2 + m_bfs[3][0]) # Sphere radius
    xc_bfs_mm = m_bfs[0][0] # Center x point
    yc_bfs_mm = m_bfs[1][0] # Center y point
    zc_bfs = m_bfs[2][0] # Center z point 

    # Get bfs map (also inliers from best model)
    z_bfs = zc_bfs - np.sqrt(R_bfs**2 - (x_mm -xc_bfs_mm)**2 - (y_mm -yc_bfs_mm)**2)
    bfs = np.zeros_like(eleMap)
    bfs[mask] = z_bfs
    #pdb.set_trace()
    #plt.figure(5);plt.imshow(bfs); plt.title('BFS'); plt.show()

    # Diff between ELE and BFS 
    dif_image = np.zeros_like(eleMap)
    dif_error = z - z_bfs
    dif_image[mask] = dif_error # original vs. predicted bfs

    #plt.figure(6);plt.imshow(dif_image); plt.title('ELE - BFS'); plt.show()

    # Fit error
    fit_error = np.abs(z - z_bfs) 
    mean_fit_error = np.mean(fit_error)

    plt.close("all")

    error_BFS = np.mean(np.abs((z - z_bfs)))

    return dif_image, error_BFS

def ransac_pac(pacMap, rng, k=10, th = 0.05, state = 0):
    """
    Params:
        pacMap - pac map
        rng - random number generator for random data partition
        s - the minimum number of image points required to fit the model
        k - the maximum number of iterations allowed in the algorithm
        th - a threshold value to determinate when a data point fits in the model
    Return:
        dif_image - difference between the best fit sphere and the examined cornea
    """

    # RANSAC variables
    iterations = 0
    s = 4
    max_inliers = 0
    p = 0.99 # Probability of having a set free of outliers
    e = 0 # Outliers ratio

    # Generate mask for pac map (estimate best map with )
    mask = pacMap > 0
    mask = binary_fill_holes(mask)
    mask_ones = mask.astype(int)

    pacMap_image = pacMap
    pacMap_image[~mask] = 0

    # plt.figure(2);plt.imshow(pacMap_image); plt.title('PAC_map'); plt.show()

    mask_est = mask

    # Get data coordinates
    px_coords = np.where(mask)
    z = pacMap[mask]
    x = px_coords[1]
    y = px_coords[0]

    # Transform into mm
    x_mm = x * 14 /141
    y_mm = y * 14 /141

    # Get data coordinates for disk area
    px_coords_est = np.where(mask_est)
    z_est = pacMap[mask_est]
    x_est = px_coords_est[1]
    y_est = px_coords_est[0]

    # Tranform into mm
    x_est_mm = x_est * 14 /141
    y_est_mm = y_est * 14 /141

    # RANSAC variables
    s_data = len(z_est)
    k = float('inf')

    # RANSAC loop
    while iterations < k:

        idxs = random_partition(s, s_data, rng)

        # Get the maybe inliers
        x_s = x_est[idxs]
        y_s = y_est[idxs]
        z_s = z_est[idxs]

        # Tranform into mm
        x_s_mm = x_s * 14 /141
        y_s_mm = y_s * 14 /141

        # Sphere fit G x m = f
        G_s = poly_matrix_RANSAC(x_s_mm, y_s_mm, z_s, s) # Asssemble G matrix
        f_s = np.zeros((len(x_s),1))
        f_s[:,0] = (x_s_mm**2) + (y_s_mm**2) + (z_s**2) #   Assemble the f vector
        m_s = np.linalg.lstsq(G_s, f_s ,rcond=None)[0] # Fit sphere with s points (maybe inliers)

        R_s = math.sqrt(m_s[0][0]**2 + m_s[1][0]**2 + m_s[2][0]**2 + m_s[3][0]) # Sphere radius
        xc_s_mm = m_s[0][0] # Center x point
        yc_s_mm = m_s[1][0] # Center y point
        zc_s = m_s[2][0] # Center z point

        distances_s = np.sqrt((y_mm - yc_s_mm)**2 + (x_mm - xc_s_mm)**2)
        max_distance_s = np.max(distances_s)

        # Get bfs map (maybe inliers)
        if R_s >= max_distance_s:
            z_bfs_maybe = zc_s - np.sqrt(R_s**2 - (x_mm -xc_s_mm)**2 - (y_mm -yc_s_mm)**2)
        else:
            continue

        bfs_maybe = np.zeros_like(pacMap)
        bfs_maybe[mask] = z_bfs_maybe

        #plt.figure(3);plt.imshow(bfs_maybe); plt.title('BFS maybe inliers'); plt.show()

        # Get only estimation points
        z_bfs_maybe_est = bfs_maybe[mask_est]

        # Calculate error
        error = np.abs(z_est - z_bfs_maybe_est)
        also_inliers = np.where(error < th)[0]
        # Evaluate which of them are also inliers
        if len(also_inliers) > max_inliers:

            # Save model with max number of inliers
            max_inliers = len(also_inliers)
            bfs_best_maybe = bfs_maybe
            idxs_maybeinliers = idxs

            # Add also inliers
            idxs_alsoinliers = also_inliers
            x_in_mm = x_est_mm[idxs_alsoinliers]
            y_in_mm = y_est_mm[idxs_alsoinliers]
            z_in = z_est[idxs_alsoinliers]

            # Update number of iterations (k)
            e = (s_data - max_inliers) / s_data
            if (1 - (1 - e) ** s) != 0:
                if( math.log(1-(1-e)**s) != 0):
                    k = (math.log(1 - p)) / (math.log(1 - (1 - e)**s))
                else:
                    k = float("inf")
            else:
                k = 0
            
        iterations += 1

    # plt.figure(4);plt.imshow(bfs_best_maybe); plt.title('BFS maybe inliers'); plt.show()

    # Estimate bfs map with the inliers of the best model
    # Sphere fit G x m = f
    G_in = poly_matrix_RANSAC(x_in_mm, y_in_mm, z_in, s) # Asssemble G matrix
    f_in = np.zeros((len(x_in_mm),1))
    f_in[:,0] = (x_in_mm**2) + (y_in_mm**2) + (z_in**2) # Asssemble f vector
    m_bfs = np.linalg.lstsq(G_in, f_in ,rcond=None)[0] # Fit sphere (also inliers from best model)
    
    R_bfs = math.sqrt(m_bfs[0][0]**2 + m_bfs[1][0]**2 + m_bfs[2][0]**2 + m_bfs[3][0]) # Sphere radius
    xc_bfs_mm = m_bfs[0][0] # Center x point
    yc_bfs_mm = m_bfs[1][0] # Center y point
    zc_bfs = m_bfs[2][0] # Center z point

    # Get bfs map (also inliers from best model)
    z_bfs = zc_bfs - np.sqrt(R_bfs**2 - (x_mm -xc_bfs_mm)**2 - (y_mm -yc_bfs_mm)**2)
    bfs = np.zeros_like(pacMap)
    bfs[mask] = z_bfs

    # plt.figure(5);plt.imshow(bfs); plt.title('BFS'); plt.show()

    # Diff between ELE and BFS
    dif_image = np.zeros_like(pacMap)
    dif_error = z - z_bfs
    dif_image[mask] = dif_error # original vs. predicted bfs

    # plt.figure(6);plt.imshow(dif_image); plt.title('ELE - BFS'); plt.show()

    # Fit error
    fit_error = np.abs(z - z_bfs)
    mean_fit_error = np.mean(fit_error)

    error_PAC = np.mean(np.mean(np.abs((z - z_bfs))))

    return dif_image, error_PAC

def random_partition(s,s_data, rng):
    """Return random idxs"""
    all_idxs = np.arange(s_data)
    rng.shuffle(all_idxs)
    idxs = all_idxs[:s]
    return idxs

def poly_matrix_RANSAC(x, y, z, s=4):
    """ Generate Matrix use with lstsq """
    ncols = s # number of columns
    G = np.zeros((x.size, ncols)) # matrix
    G[:, 0] = x*2
    G[:, 1] = y*2
    G[:, 2] = z*2
    G[:, 3] = 1
    return G

def poly_matrix(x, y, z, order = 2):
    """ Generate Matrix use with lstsq """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols)) # matrix
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij): 
        G[:,k] = x**i * y**j
    return G

def quadraticFit(image, pacimage, diskRadius=0):
   
    # Create mask for ELE map
    mask=image>0
    mask=binary_fill_holes(mask)
    mask_ones=np.zeros_like(mask)
    mask_ones[mask]=1
    
    # Get minimum point from PAC image
    if diskRadius>0:
        
        minPoint=getMinPointPac(pacimage) 
        mask_removal = np.zeros_like(mask)
        mask_removal[minPoint]=1 
        mask_removal=binary_dilation(mask_removal,footprint=disk(diskRadius)) 
        
        mask_alive = np.logical_not(mask_removal) 
        mask_est=np.logical_and(mask,mask_alive) 

    else:
        mask_est=mask
    
    ordr = 2  # order of polynomial
    px_coords = np.where(mask_est) 
    z=image[mask_est] 

    x = px_coords[1]
    y = px_coords[0] 

    meanx=x.mean()
    meany=y.mean()

    stdx=x.std()
    stdy=y.std()

    # Normalization
    x = (x - meanx)/stdx
    y = (y - meany)/stdy# this improves accuracy
    
    # Solve problem
    G = poly_matrix(x, y, ordr)
    # # Solve for np.dot(G, m) = z:
    m = np.linalg.lstsq(G, z,rcond=None)[0] # Result m --> G m = z

    z=image[mask]
    px_coords = np.where(mask)
    x = px_coords[1]
    y = px_coords[0]
    x = (x - meanx)/stdx
    y = (y - meany)/stdy # this improves accuracy
    G = poly_matrix(x, y, ordr) 
    
    # BFS map
    zz = np.reshape(np.dot(G, m), x.shape)
    bfsimage= np.zeros_like(image)
    bfsimage[mask]=zz

    dif_image=np.zeros_like(image) 
    dif_image[mask]= z - zz # original vs. the predicted sphere 
    
    return dif_image

def getMinPointPac(pacimage):

    mask=pacimage>=0
    mask=binary_erosion(mask, footprint=disk(5))
    mask=binary_fill_holes(mask)
    pacimage[mask==0]=10000000
    
    #Sometimes, there are more than one point with the same density => We low-pass filter the image to be more sure
    px_coords = np.unravel_index(np.argmin(pacimage, axis=None), pacimage.shape)
    minVal = pacimage[px_coords[0],px_coords[1]] 
    
    if (pacimage==minVal).sum()>1:
        sigma=0.5
        while 1:
            pacf=gaussian(pacimage, sigma=sigma)
            pacf[mask==0]=100000000
            px_coords = np.unravel_index(np.argmin(pacf, axis=None), pacimage.shape)
            minVal = pacf[px_coords[0],px_coords[1]] 
            if (pacf==minVal).sum()==1:
                break
            sigma=sigma*2
    return px_coords
  
def createIndexFile_ssl(patient_idx,df_patients, name, session_index_OS, session_index_OD):
    
    split = name.split('_')
    sesion = split[-3]
    ojo = split[-4]
    patient_name = ' '.join([split[0], split[1]])
    if ojo == 'OS':
        
        if session_index_OS == 0:
            temp = 1
            session_index_OS = 8
        else:
            temp = 2
            sesion_index_OS = 0
    else:
        
        if session_index_OD == 0:
            temp = 1
            session_index_OD = 8
        else:
            temp = 2
            sesion_index_OD = 0
        
    new_patient = [patient_name,patient_idx,ojo,str(sesion), temp, 'multicentrico']
    
    return new_patient, session_index_OS, session_index_OD

def createIndexFile(fullAnnFile,annFile,data_path,biomarkers=False):
    
    columns=["Número de Registro","Lado (OD 0 OI 1)","Hospital","Descompensación corneal","INCLUIDOS EN ESTUDIO"]
    if biomarkers:
        columns = columns + ["Paq Relativa","Densitometria ant"]
    
    df = pd.read_csv(fullAnnFile, delimiter=';',usecols=columns)
    df=df[df['INCLUIDOS EN ESTUDIO']=='0']
    df = df.iloc[2:]
    print(df.head(5))
    
    df= df.dropna()
    df["Lado (OD 0 OI 1)"]=df["Lado (OD 0 OI 1)"].astype(int)
    df["Descompensación corneal"]=df["Descompensación corneal"].astype(float).astype(int)
    # df= df.astype(int)
    print(df.isnull().sum().sum())

    # using len(df) check is the dataframe is empty
    print(len(df) == 0)
    
    #create training images annotation file ### shouldn't do that for all images at the begining?
    columns = ["img_id","img_path","db","hospital","lado","label"]
    if biomarkers:
        columns = columns + ["PaqRel","DensAnt"]
    
    new_train_df = pd.DataFrame(columns=columns)
    new_train_df["img_id"] = df["Número de Registro"];
    new_train_df["db"] = "multicentrico"
    new_train_df["hospital"] = df["Hospital"].str.upper()
    new_train_df["img_path"] = data_path + '/' + df["Número de Registro"];
    new_train_df["lado"] = df["Lado (OD 0 OI 1)"];
    new_train_df["label"] = df["Descompensación corneal"];
    if biomarkers:
        new_train_df["PaqRel"] = df["Paq Relativa"].astype(float);
        new_train_df["DensAnt"] = df["Densitometria ant"].astype(float);
    
    new_train_df.to_csv(annFile,index=False)
    
    return new_train_df
          
def computeNorm(annFile, data_path, px = 100):
    
    # dataset = pd.read_csv(annFile,header=0,dtype={'img_id': str, 'label': int})
    dataset = pd.read_csv(annFile,header=0, dtype={'Sesion': str})

    #To check the different files
    listFiles=glob.glob(os.path.join(data_path,'1_*.npy'))
    categories=[os.path.basename(fichero).split('_')[1] + '_' + os.path.basename(fichero).split('_')[4] [:-4] for fichero in listFiles]
    mom1 = np.zeros(len(categories),)
    mom2 = np.zeros(len(categories),)
    cont = np.zeros(len(categories),)
    for i in range(len(dataset)):

        index = dataset ['Patient'].values[i]
        eye = dataset ['Ojo'].values[i]
        session = dataset ['Sesion'].values[i]

        #Calculate mean and std for every map category
        for c in range(len(categories)):
            split_cat = categories[c].split('_')
            impath = str(index) + '_' + split_cat[0] + '_' + eye + '_' + str(session) + '_' + split_cat[1] + '.npy'
            im = np.load(data_path + '/' + impath)
            if im is None:
                pdb.set_trace()
            (h,w) = im.shape
            mask = im > - px
            
            #Files
            mom1[c] += im[mask].sum()
            mom2[c] += (im[mask]**2).sum()
            cont[c] += mask.sum()
    mean = mom1/cont
    mom2 = mom2/cont
    
    sigma = np.sqrt(mom2-mean**2)
    return categories, mean, sigma

if __name__ == "__main__":
    """ Arguments
        dir: results directory
        th: RANSAC threshold
        r: estimation disk radius in px
        ransac: 
            - -1: original maps
            - 0:  BFS estimation with quadraticfit (baseline method)
            - 1:  BFS estimation with RANSAC
        pixels: background pixels value
    """
    eConfig = {
        'dir':'RESNET',
        'th':'1.5',
        'r':'45', 
        'ransac':'1',
        'pixels':'100'
        }
    # Arguments
    args = sys.argv[1::]
    for i in range(0,len(args),2):
        key = args[i]
        val = args[i+1]
        eConfig[key] = type(eConfig[key])(val)

    # Random generator
    seed = 0

    map_cats = ['PAC','CUR','ELE','CORNEA-DENS']
    db_path = '../datasets/dataset_multicentrico'
    csv_path = db_path + '/csv'

    ransac = int(eConfig['ransac']) # 0 --> quadraticFit (baseline), 1 --> RANSAC, -1 --> original
    pixels = int(eConfig['pixels']) # pixels background value

    # Folders to save data
    data_path = db_path + '/data' 
    image_path = db_path + '/images'
    imageHealthy_path = image_path + '/healthy'
    imageSick_path = image_path + '/sick'
    imageNolabel_path = image_path + '/noLabel'

    heat_path = db_path + '/heatMaps'
    heatHealthy_path = heat_path + '/healthy'
    heatSick_path = heat_path + '/sick'
    heatNolabel_path = heat_path + '/noLabel'

    polar_path = db_path + '/polarMaps'
    polarHealthy_path = polar_path + '/healthy'
    polarSick_path = polar_path + '/sick'
    polarNolabel_path = polar_path + '/noLabel'

    error_BFS_path = db_path + '/error_BFS' 
    error_PAC_path = db_path + '/error_PAC' 

    list_path = db_path+'/patient_list.csv'

    # Name of the annotation files 
    fullAnnFile = db_path + '/full_annotation.csv'
    annFile = db_path + '/annotation.csv'
    annBioFile = db_path + '/annotation_biomarkers.csv'
    df_patients = pd.DataFrame(columns=['Nombre', 'Patient','Ojo', 'Sesion', 'Temp', 'Database'])
    

    # Create new directory 
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    if not os.path.exists(imageHealthy_path):
        os.makedirs(imageHealthy_path)

    if not os.path.exists(imageSick_path):
        os.makedirs(imageSick_path)

    if not os.path.exists(imageNolabel_path):
        os.makedirs(imageNolabel_path)

    if not os.path.exists(heat_path):
        os.makedirs(heat_path)

    if not os.path.exists(heatHealthy_path):
        os.makedirs(heatHealthy_path)

    if not os.path.exists(heatSick_path):
        os.makedirs(heatSick_path)

    if not os.path.exists(heatNolabel_path):
        os.makedirs(heatNolabel_path)
    
    if not os.path.exists(polar_path):
        os.makedirs(polar_path)

    if not os.path.exists(polarHealthy_path):
        os.makedirs(polarHealthy_path)

    if not os.path.exists(polarSick_path):
        os.makedirs(polarSick_path)

    if not os.path.exists(polarNolabel_path):
        os.makedirs(polarNolabel_path)

    if not os.path.exists(error_PAC_path):
        os.makedirs(error_PAC_path)
    
    if not os.path.exists(error_BFS_path):
        os.makedirs(error_BFS_path)


    # # Create index file
    db_index = createIndexFile(fullAnnFile,annFile,data_path,biomarkers=False)

    # # Create index file for experiments comparing Arnalich, 2019 (reduced dataset)
    createIndexFile(fullAnnFile,annBioFile,data_path,biomarkers=True)

    info_dataset = pd.read_csv(annFile, header=0, dtype={'img_id': str, 'label': int}) #dictionary variable

    # First calculate max and min for all map categories
    print('-'*5 + 'COMPUTE MAX AND MIN FOR ALL MAPS' + '-'*5)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    for patient, hospital, label in zip(db_index['img_id'], db_index['hospital'], db_index['label']):
        base_folder = os.path.join(csv_path, hospital, 'DATOS EXPORTADOS', patient)
        basal_folder = find_folder(base_folder, 'BASAL')
        tras_faco_folder = find_folder(base_folder, 'TRAS FACO')
        if (basal_folder == None or tras_faco_folder == None):
            continue
        basal_csv_files = glob.glob(basal_folder + '/*PAC.CSV')
        tras_faco_csv_files = glob.glob(tras_faco_folder + '/*PAC.CSV')
        
        basal_patient_path= [basal_folder for file in basal_csv_files]
        tras_faco_patient_path= [tras_faco_folder for file in tras_faco_csv_files]

        csv_files = basal_csv_files + tras_faco_csv_files
        patients_path = basal_patient_path + tras_faco_patient_path

        session_index_OS = 0
        session_index_OD = 0

        # comprobación de que los mapas tienen su pareja y tienen todos los mapas
        list_maps_basal = glob.glob(basal_folder + '/*.CSV')
        categories_basal = [os.path.basename(fichero).split('_')[-1][:-4] for fichero in list_maps_basal]
    
        list_maps_tras_faco = glob.glob(tras_faco_folder + '/*.CSV')
        categories_tras_faco = [os.path.basename(fichero).split('_')[-1][:-4]  for fichero in list_maps_tras_faco]
    
        if not set(map_cats).issubset(set(categories_basal)) or not set(map_cats).issubset(set(categories_tras_faco)):
            continue

        for f, patient_path in zip_longest(csv_files, patients_path):

            for map_cat in map_cats:

                if map_cat != 'PAC':
                
                    if re.search(r'_\d{1}b_|_\d{1}_', f):
                        f = re.sub(r'_\d{1}b_|_\d{1}_', '', f)
                
                csv_name = f.split('/')[-1].replace('PAC',map_cat)
                name = csv_name.replace(".CSV",'')

                split = name.split('_')
                sesion = split[-3]
                ojo = split[-4]

                outname_short = patient + '_' + map_cat 
                outname = patient + '_' + map_cat + '_' + ojo + '_' + sesion 
                if not os.path.exists(patient_path + '/' + csv_name):
                    print("No ENCONTRADO: " + patient_path + '/' + csv_name)
                    continue
                dataset = readCSV(patient_path,csv_name)
                computeMaxMin(dataset, outname_short, outname )
            
    print('PAC')
    print('max = ' + str(PAC_max))
    print('min = ' + str(PAC_min))
    
    for i in np.arange(len(ELE_max)):
        print('ELE[{}]'.format(i))
        print('max = ' + str(ELE_max[i]))
        print('min = ' + str(ELE_min[i]))
    
    for i in np.arange(len(CUR_max)):
        print('CUR[{}]'.format(i))
        print('max = ' + str(CUR_max[i]))
        print('min = ' + str(CUR_min[i]))

    for i in np.arange(len(CORNEA_DENS_max)):
        print('CORNEA_DENS[{}]'.format(i))
        print('max = ' + str(CORNEA_DENS_max[i]))
        print('min = ' + str(CORNEA_DENS_min[i]))
    
    for i in np.arange(len(ELE_BFS_8mm_max)):
        print('ELE_BFS_8mm[{}]'.format(i))
        print('max = ' + str(ELE_BFS_8mm_max[i]))
        print('min = ' + str(ELE_BFS_8mm_min[i]))
    
    # Calculate max and min for BFS maps
    print('-'*5 + 'COMPUTE MAX AND MIN FOR BFS MAPS' + '-'*5)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    for patient, hospital, label in zip(db_index['img_id'], db_index['hospital'], db_index['label']):
        base_folder = os.path.join(csv_path, hospital, 'DATOS EXPORTADOS', patient)
        basal_folder = find_folder(base_folder, 'BASAL')
        tras_faco_folder = find_folder(base_folder, 'TRAS FACO')
        if (basal_folder == None or tras_faco_folder == None):
            continue
        basal_csv_files = glob.glob(basal_folder + '/*PAC.CSV')
        tras_faco_csv_files = glob.glob(tras_faco_folder + '/*PAC.CSV')
        
        basal_patient_path= [basal_folder for file in basal_csv_files]
        tras_faco_patient_path= [tras_faco_folder for file in tras_faco_csv_files]

        csv_files = basal_csv_files + tras_faco_csv_files
        patients_path = basal_patient_path + tras_faco_patient_path

        session_index_OS = 0
        session_index_OD = 0
        
        # comprobación de que los mapas tienen su pareja y tienen todos los mapas
        list_maps_basal = glob.glob(basal_folder + '/*.CSV')
        categories_basal = [os.path.basename(fichero).split('_')[-1][:-4] for fichero in list_maps_basal]
    
        list_maps_tras_faco = glob.glob(tras_faco_folder + '/*.CSV')
        categories_tras_faco = [os.path.basename(fichero).split('_')[-1][:-4]  for fichero in list_maps_tras_faco]
    
        if not set(map_cats).issubset(set(categories_basal)) or not set(map_cats).issubset(set(categories_tras_faco)):
            continue

        for f, patient_path in zip_longest(csv_files, patients_path):

            for map_cat in map_cats:

                if map_cat != 'PAC':
                
                    if re.search(r'_\d{1}b_|_\d{1}_', f):
                        f = re.sub(r'_\d{1}b_|_\d{1}_', '', f)
                
                csv_name = f.split('/')[-1].replace('PAC',map_cat)
                name = csv_name.replace(".CSV",'')

                split = name.split('_')
                sesion = split[-3]
                ojo = split[-4]

                outname_short = patient + '_' + map_cat 
                outname = patient + '_' + map_cat + '_' + ojo + '_' + sesion 
                if not os.path.exists(patient_path + '/' + csv_name):
                    print("No ENCONTRADO: " + patient_path + '/' + csv_name)
                    continue
                dataset = readCSV(patient_path,csv_name)
                computeMaxMinBFS(data_path, dataset, outname_short, outname, info_dataset, rng, th = float(eConfig['th']), diskRadius= int(eConfig['r']), ransac = ransac, px = pixels)
        
    for i in np.arange(len(ELE_BFS_max)):
        print('ELE_BFS[{}]'.format(i))
        print('max = ' + str(ELE_BFS_max[i]))
        print('min = ' + str(ELE_BFS_min[i]))  
    
    # Generate data
    print('-'*5 + 'DATA GENERATION' + '-'*5)
       # Loop of patients
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    for patient, hospital, label in zip(db_index['img_id'], db_index['hospital'], db_index['label']):
        base_folder = os.path.join(csv_path, hospital, 'DATOS EXPORTADOS', patient)
        basal_folder = find_folder(base_folder, 'BASAL')
        tras_faco_folder = find_folder(base_folder, 'TRAS FACO')
        if (basal_folder == None or tras_faco_folder == None):
            continue
        basal_csv_files = glob.glob(basal_folder + '/*PAC.CSV')
        tras_faco_csv_files = glob.glob(tras_faco_folder + '/*PAC.CSV')
        
        basal_patient_path= [basal_folder for file in basal_csv_files]
        tras_faco_patient_path= [tras_faco_folder for file in tras_faco_csv_files]

        csv_files = basal_csv_files + tras_faco_csv_files
        patients_path = basal_patient_path + tras_faco_patient_path

        session_index_OS = 0
        session_index_OD = 0
        
        # comprobación de que los mapas tienen su pareja y tienen todos los mapas
        list_maps_basal = glob.glob(basal_folder + '/*.CSV')
        categories_basal = [os.path.basename(fichero).split('_')[-1][:-4] for fichero in list_maps_basal]
    
        list_maps_tras_faco = glob.glob(tras_faco_folder + '/*.CSV')
        categories_tras_faco = [os.path.basename(fichero).split('_')[-1][:-4]  for fichero in list_maps_tras_faco]
    
        if not set(map_cats).issubset(set(categories_basal)) or not set(map_cats).issubset(set(categories_tras_faco)):
            continue

        for f, patient_path in zip_longest(csv_files, patients_path):

            for map_cat in map_cats:

                if map_cat != 'PAC':
                
                    if re.search(r'_\d{1}b_|_\d{1}_', f):
                        f = re.sub(r'_\d{1}b_|_\d{1}_', '', f)
                
                csv_name = f.split('/')[-1].replace('PAC',map_cat)
                name = csv_name.replace(".CSV",'')

                split = name.split('_')
                sesion = split[-3]
                ojo = split[-4]

                outname_short = patient + '_' + map_cat 
                outname = patient + '_' + map_cat + '_' + ojo + '_' + sesion 
                if not os.path.exists(patient_path + '/' + csv_name):
                    print("No ENCONTRADO: " + patient_path + '/' + csv_name)
                    continue
                dataset = readCSV(patient_path,csv_name) 
                print(outname)
                dataGenerator(data_path, error_BFS_path, error_PAC_path, image_path, heat_path, polar_path, dataset, outname_short, outname, info_dataset, rng, th = float(eConfig['th']), diskRadius= int(eConfig['r']), ransac = ransac, px = pixels)
            new_patient, session_index_OS, session_index_OD = createIndexFile_ssl(patient,df_patients,name, session_index_OS, session_index_OD)
            df_patients.loc[len(df_patients)] = new_patient
    # Normalization
    df_patients.to_csv(list_path,index = False)
    cats,mu,sigma = computeNorm(list_path, data_path, px = pixels)
    normalization_data = {'categories': cats, 'mean': mu, 'std': sigma}
    print(normalization_data)

    normalization_file = os.path.join(data_path,'normalization.npz')
    with open(normalization_file, 'wb') as f:
        pickle.dump(normalization_data, f)