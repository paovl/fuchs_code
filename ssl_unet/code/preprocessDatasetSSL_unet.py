#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:40:01 2024

@author: mariagonzalezgarcia
"""

import numpy as np
import pandas as pd
import os
import glob
import pathlib
import PIL
from PIL import Image
import pandas as pd
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as img
import pdb
from skimage.transform import AffineTransform, warp
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_erosion, binary_dilation, disk
from skimage.filters import gaussian
import itertools
import argparse
import sys
import re
from sklearn.preprocessing import label_binarize
from skimage.measure import label as label_mask
from skimage.measure import label, regionprops
import math
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

error_BFS_map_max = np.zeros((6,))
error_BFS_map_min = np.full((6,), float('inf'))

error_PAC_map_max = np.zeros((6,))
error_PAC_map_min = np.full((6,), float('inf'))

error_CD_map_max = np.zeros((6,))
error_CD_map_min = np.full((6,), float('inf'))

CORNEA_DENS_max = np.zeros((2,))
CORNEA_DENS_min = np.full((2,), float('inf'))

def parse_args():
    """
    Parse input arguments
    
    """
    
    parser = argparse.ArgumentParser(description='Script for data preprocessing')
    parser.add_argument('--preprocesspath', dest='path_prepro',
                        help='name the path to preprocess the data. Default = ../datasets/dataset_ssl',
                        default='../datasets/dataset_ssl', type=str)
                        
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def readCSV(csv_path, csv):
    
    """ Read csv files """
    
    df = pd.read_csv( os.path.join(csv_path,csv), delimiter=';',
                     decimal=',',encoding='unicode_escape')
    df = df.fillna(0)
    
    # obtain the index of the row start with '[system]' and drop the rest of the rows
    end = df[df.iloc[:, 0] == '[SYSTEM]'].index.tolist()[0]
    dataset = df.iloc[:end]
    dataset = dataset.replace(',', '.', regex=True)
    
    return dataset

def computeMaxMin(dataset, fname, outname):
    """
    This functions goes through all maps in order to get max and min for each map category
    """

    global PAC_max, PAC_min, CUR_max, CUR_min, ELE_max, ELE_min, CORNEA_DENS_max, CORNEA_DENS_min

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

        mask = pixels > 0  # Gets a mask for pixels with greater values than 0
        mask = binary_fill_holes(mask)
        
        max_value = pixels[mask].max()
        min_value = pixels[mask].min()

        if fname.find('ELE') > 0:

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
                
def computeMaxMinBFS(data_path, dataset, fname, outname, rng,  th = 1.5, diskRadius= 45, ransac = 1, px = 100):
    """
    This functions goes through all ELE_3 and ELE_4 maps in order to get max and min values for the BFS maps
    """

    global PAC_max, PAC_min, ELE_max, ELE_min, ELE_BFS_max, ELE_BFS_min, error_BFS_map_max, error_BFS_map_min, error_PAC_map_max, error_PAC_map_min, error_CD_map_max, error_CD_map_min

    # Aux variables
    i = 0
    j = 0

    step = 141 # Image rows
    iterations = int(dataset.shape[0]/step) # Count rows in the file

    outname_split = fname.split('_')
    session = outname_split[-3]
    # outname = '%s_%s'%(outname_split[0],outname_split[-1])

    od = fname.find('_OD_') > 0 # OD (right eye)
    ele = fname.find('_ELE') > 0 # ELE MAP
    pac = fname.find('_PAC') > 0 # PAC MAP
    cd = fname.find('CORNEA-DENS') > 0 # CD MAP

    #Processes every image from the dataset
 
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
            _, _, error_PAC_map = ransac_pac(pixels, rng, th = th * 1000, diskRadius = diskRadius, state = 1)

            min_error_map_value = error_PAC_map[mask].min()
            max_error_map_value = error_PAC_map[mask].max()

            if (max_error_map_value > error_PAC_map_max[k]):
                error_PAC_map_max[k] = max_error_map_value
            if (min_error_map_value < error_PAC_map_min[k]):
                error_PAC_map_min[k] = min_error_map_value
            
            pixels = (pixels - PAC_min) / (PAC_max - PAC_min)
            # Save data
            pixels[~mask] = -px
    
            np.save(os.path.join(data_path,outname + '_' + str(k) + '.npy'), pixels)
            
        if ele and ransac != -1 and k == 4 :
           
            # RANSAC algorithm with sphere fit
            if ransac == 1:
                pixels, _, error_BFS_map = ransac_bfs(pixels, rng, th = th, diskRadius = diskRadius, state = 1)

                max_error_map_value = error_BFS_map[mask].max()
                min_error_map_value = error_BFS_map[mask].min()

                if (max_error_map_value > error_BFS_map_max[k]):
                    error_BFS_map_max[k] = max_error_map_value

                if (min_error_map_value < error_BFS_map_min[k]):
                    error_BFS_map_min[k] = min_error_map_value

            # No RANSAC with quadratic fit
            if ransac == 0:
                if od:
                    pacimage = np.load(os.path.join(data_path,outname.replace('ELE','PAC') + '_0.npy')) # Recover PAC map
                else:
                    pacimage = np.load(os.path.join(data_path,outname.replace('ELE','PAC') + '_0.npy')) # Recover PAC map
                pixels_rel = quadraticFit(pixels, pacimage, diskRadius = diskRadius)
                pixels = pixels_rel

            max_value = pixels[mask].max()
            min_value = pixels[mask].min()

            if(max_value > ELE_BFS_max[k]):
                ELE_BFS_max[k] = max_value

            if(min_value < ELE_BFS_min[k]):
                ELE_BFS_min[k] = min_value
        if cd:
            _, _, error_CD_map = ransac_cd(pixels, rng, th = th * (100/3.5), diskRadius = diskRadius, state = 1)

            min_error_map_value = error_CD_map[mask].min()
            max_error_map_value = error_CD_map[mask].max()

            if (max_error_map_value > error_CD_map_max[k]):
                error_CD_map_max[k] = max_error_map_value
            if (min_error_map_value < error_CD_map_min[k]):
                error_CD_map_min[k] = min_error_map_value

def heatMapGenerator(fpath,pixels_img,od,k,session,outname):
    
    """ Generate heatmap images of each topographic map for visualization """

    if od:
        image = img.imsave(os.path.join(fpath,outname + '_OD_'+ session + '_'+ str(k) + '.png'), pixels_img.astype(np.float64),cmap = 'jet')
    else:
        image = img.imsave(os.path.join(fpath,outname + '_OS_' + session + '_'+ str(k) + '.png'), pixels_img.astype(np.float64),cmap = 'jet')

def poly_matrix(x, y, order=2):

    """ Generate Matrix use with lstsq """
    
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    
    return G

def getMinPointPac(pacimage):
    # plt.figure();plt.imshow(pac,cmap='gray')
    mask=pacimage>0
    # plt.figure();plt.imshow(mask,cmap='gray')
    mask=binary_erosion(mask, footprint=disk(5))
    # plt.figure();plt.imshow(mask,cmap='gray')
    mask=binary_fill_holes(mask)
    # plt.figure();plt.imshow(mask,cmap='gray')
    pacimage[mask==0]=10000000
    # plt.figure();plt.imshow(pac,cmap='gray')
    
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
    
def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = np.sqrt(t)

    return radius, C[0], C[1], C[2]

def sphereProject(radius,C,x,y):
    
    zzo=np.sqrt(radius**2 - (x-C[0])**2 - (y-C[1])**2 )
    zz=zzo+C[2]
    return zz

def quadraticFit(image,pacimage,diskRemoval=0):
    # plt.figure(1);plt.imshow(image);
    # pdb.set_trace()
    ordr = 2  # order of polynomial
    mask=image>0;
    # mask=binary_erosion(mask,footprint=disk(3))
    mask=binary_fill_holes(mask)
    # plt.figure(2);plt.imshow(mask);
    
    #get minimum point
    # plt.figure(3);plt.imshow(pacimage);
    # pdb.set_trace()
    if diskRemoval>0:
        minPoint=getMinPointPac(pacimage)
        mask_removal = np.zeros_like(mask)
        mask_removal[minPoint]=1
        mask_removal=binary_dilation(mask_removal,footprint=disk(diskRemoval))
        # plt.figure(4);plt.imshow(mask_removal);
        mask_alive = np.logical_not(mask_removal)
        mask_est=np.logical_and(mask,mask_alive);
        # plt.figure(5);plt.imshow(mask_est);
    else:
        mask_est=mask;
    #Now we hace to remove the exclusion area
    #First we identify the minium
    px_coords = np.where(mask_est)
    z=image[mask_est]
    x = px_coords[1]
    y = px_coords[0]
    meanx=x.mean()
    meany=y.mean()
    stdx=x.std()
    stdy=y.std()
    
    # pdb.set_trace()
    x = (x - meanx)/stdx
    y = (y - meany)/stdy# this improves accuracy
    
    # Solve problem
    G = poly_matrix(x, y, order = ordr)
    # # Solve for np.dot(G, m) = z:
    m = np.linalg.lstsq(G, z,rcond=None)[0]
    # pdb.set_trace()
    
    # radius, x0, y0, z0 = sphereFit(x,y,z)
    #Now generate the new polymatrix
    z=image[mask]
    px_coords = np.where(mask)
    x = px_coords[1]
    y = px_coords[0]
    x = (x - meanx)/stdx
    y = (y - meany)/stdy# this improves accuracy
    
    G = poly_matrix(x, y, order = ordr)
    zz = np.reshape(np.dot(G, m), x.shape)
    
    # zz = sphereProject(radius,(x0, y0, z0),x,y)
    # pdb.set_trace()
    # difz=z-zz
    dif_image=np.zeros_like(image)
    dif_image[mask]=z-zz
    
    # print((dif_image[mask].min(),dif_image[mask].max()))
    
    # plt.figure(6);plt.imshow(dif_image);plt.show()
    # plt.figure();plt.imshow(image);
    # plt.figure();plt.imshow(dif_image);
    # plt.show()
    # pdb.set_trace()
    return dif_image

def ransac_bfs(eleMap, rng, k=10, th = 0.05, diskRadius= 0, state = 0):
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

    # RANSAC variables
    iterations = 0
    s = 4
    max_inliers = 0
    p = 0.99 # Probability of having a set free of outliers
    e = 0 # Outliers ratio

    # Generate mask for ele map
    mask = eleMap > 0
    mask = binary_fill_holes(mask)
    mask_ones = mask.astype(int)

    # plt.figure(1);plt.imshow(eleMap); plt.title('ELE map'); plt.show() # plot ELE map

    # Disk used to estimate BFS map
    if diskRadius> 0:
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

    # Transform into mm
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

    #plt.figure(5);plt.imshow(bfs); plt.title('BFS'); plt.show()

    # Diff between ELE and BFS
    dif_image = np.zeros_like(eleMap)
    dif_error = z - z_bfs
    dif_image[mask] = dif_error # original vs. predicted bfs

    #plt.figure(6);plt.imshow(dif_image); plt.title('ELE - BFS'); plt.show()

    # Fit error
    fit_error = np.abs(z - z_bfs)
    mean_fit_error = np.mean(fit_error)

    error_BFS = np.mean(np.abs((z - z_bfs)))

    error_BFS_map = np.zeros_like(eleMap)
    # Me quedo con todos (mayor distancia que BFS y menor distancia que BFS)
    error_BFS_map[mask] = np.abs((z - z_bfs))
    # Me quedo solo con los positivos (mayor distancia que BFS (estan por debajo))
    # error_BFS_map[mask] = (z - z_bfs)
    # error_BFS_map[error_BFS_map < 0] = 0
    # error_BFS_map = np.abs(error_BFS_map)

    return dif_image, error_BFS, error_BFS_map

def ransac_pac(pacMap, rng, k=10, th = 0.05, diskRadius = 0, state = 0):
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

    # plt.figure(2);plt.imshow(pacMap_image); plt.title('PAC_map'); plt.show()

    mask_est = mask

    # Get data coordinates
    px_coords = np.where(mask)
    z = pacMap[mask]
    x = px_coords[1]
    y = px_coords[0]

    # Transform into mm
    x_mm = x * 14000 /141
    y_mm = y * 14000 /141

    # Get data coordinates for disk area
    px_coords_est = np.where(mask_est)
    z_est = pacMap[mask_est]
    x_est = px_coords_est[1]
    y_est = px_coords_est[0]

    # Tranform into mm
    x_est_mm = x_est * 14000 /141
    y_est_mm = y_est * 14000 /141

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
        x_s_mm = x_s * 14000 /141
        y_s_mm = y_s * 14000 /141

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

    # Me quedo con ambos (mayor y menos grosor)
    error_PAC_map = np.zeros_like(pacMap)
    # error_PAC_map[mask] = np.abs((z - z_bfs))
    # Me quedo solo con (mayor grosor)
    error_PAC_map[mask] = (z - z_bfs)
    error_PAC_map[error_PAC_map < 0] = 0
    error_PAC_map = np.abs(error_PAC_map)

    return dif_image, error_PAC, error_PAC_map

def ransac_cd(cdMap, rng,  k=10, th = 0.05, diskRadius= 0, state = 0):

    # RANSAC variables
    iterations = 0 
    s = 9
    max_inliers = 0 
    p = 0.99 # Probability of having a set free of outliers
    e = 0 # Outliers ratio
    ordr = 2

    # Generate mask for ele map
    mask = cdMap > 0 # Boolean map
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
    z = cdMap[mask] 
    x = px_coords[1]
    y = px_coords[0] 

    # Tranform into mm
    x_mm = (x - x.mean()) / x.std()
    y_mm = (y - y.mean()) / y.std()

    # Get data coordinates for disk area 
    px_coords_est = np.where(mask_est) 
    z_est = cdMap[mask_est]
    x_est = px_coords_est[1]
    y_est = px_coords_est[0]

    # Tranform into mm
    x_est_mm = (x_est - x.mean()) / x.std()
    y_est_mm = (y_est - y.mean()) / y.std()

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
        x_s_mm = (x_s - x.mean()) / x.std()
        y_s_mm = (y_s - y.mean()) / y.std()

        # Sphere fit G x m = f
        G_s = poly_matrix(x_s_mm, y_s_mm, order = ordr)
        m_s = np.linalg.lstsq(G_s, z_s ,rcond=None)[0] # Fit sphere with s points (maybe inliers)
        G_s_maybe = poly_matrix(x_mm, y_mm, order = ordr) # Asssemble G matrix
        
        z_bfs_maybe = np.dot(G_s_maybe, m_s) # Get bfs map (maybe inliers)
        bfs_maybe = np.zeros_like(cdMap)
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
    G_in = poly_matrix(x_in_mm, y_in_mm, order = ordr) # Asssemble G matrix 
    m_bfs = np.linalg.lstsq(G_in, z_in ,rcond=None)[0] # Fit sphere (also inliers from best model)
    G_bfs = poly_matrix(x_mm, y_mm, order = ordr) # Asssemble G matrix

    z_bfs = np.dot(G_bfs, m_bfs) # Get bfs map (also inliers from best model)
    bfs = np.zeros_like(cdMap)
    bfs[mask] = z_bfs
    #pdb.set_trace()
    #plt.figure(5);plt.imshow(bfs); plt.title('BFS'); plt.show()

    # Diff between ELE and BFS 
    dif_image = np.zeros_like(cdMap)
    dif_error = z - z_bfs
    dif_image[mask] = dif_error # original vs. predicted bfs

    #plt.figure(6);plt.imshow(dif_image); plt.title('ELE - BFS'); plt.show()

    # Fit error
    fit_error = np.abs(z - z_bfs) 
    mean_fit_error = np.mean(fit_error)

    plt.close("all")

    error_CD = np.mean(np.abs((z - z_bfs)))

    error_CD_map = np.zeros_like(cdMap)
    # Me quedo con ambos (mayor y menos opacidad)
    # error_CD_map[mask] = np.abs((z - z_bfs))
    # Me quedo solo con valores positivos (mayor opacidad)
    error_CD_map[mask] = (z - z_bfs)
    error_CD_map[error_CD_map < 0] = 0
    error_CD_map = np.abs(error_CD_map)

    return dif_image, error_CD, error_CD_map

def random_partition(s, s_data, rng):
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
    
def imageGenerator(fpath,pixels_img, mask, od, k, session, outname):
        
    if od:
        image = cv2.imwrite(os.path.join(fpath,outname + '_OD_' + session + '_'+ str(k) + '.png'), np.uint16(65535*pixels_img))
    else:
        image = cv2.imwrite(os.path.join(fpath,outname + '_OS_' +  session + '_'+ str(k) + '.png'), np.uint16(65535*pixels_img))

def dataGenerator(data_path, error_BFS_path, error_PAC_path, error_CD_path, image_path, heatmap_path, polar_image_path, dataset,fname, outname, patient, rng, th = 1.5, diskRadius=45, ransac = 1, px=100):

    global PAC_max, PAC_min, CUR_max, CUR_min, ELE_max, ELE_min, CORNEA_DENS_max, CORNEA_DENS_min, ELE_BFS_max, ELE_BFS_min, error_BFS_map_max, error_BFS_map_min, error_PAC_map_max, error_PAC_map_min, error_CD_map_max, error_CD_map_min

    i = 0
    j = 0
    step = 141 # rows of an image
    iteracion = int(dataset.shape[0]/step) # count the images in a file
    outname_split=fname.split('_')
    session = outname_split[-3]
    od = fname.find('_OD_')>0
    ele = fname.find('_ELE')>0
    pac = fname.find('_PAC')>0
    cd = fname.find('CORNEA-DENS') > 0 # CD map

    for k in range(iteracion):
        j = i + step
        array = dataset[i:j].to_numpy()
        pixels = array[:,1:]
        i = j + 1 
        pixels=pixels.astype(np.float64)
        if od:  
            pixels = pixels[:-1:]
        mask=pixels>0;  
        mask=binary_fill_holes(mask)
        
        # Normalization

        if fname.find('ELE') > 0 and (ransac == -1 or k != 4) :
            pixels = (pixels - ELE_min[k]) / (ELE_max[k] - ELE_min[k])

        elif fname.find('CUR') > 0:
            pixels = (pixels - CUR_min[k]) / (CUR_max[k] - CUR_min[k])

        if pac:

            _, error_PAC, error_PAC_map  = ransac_pac(pixels, rng, th = th * 1000, diskRadius= diskRadius, state = 1)
            
            np.save(os.path.join(error_PAC_path,outname + '_' + str(k) + '.npy'), error_PAC)
            
            error_PAC_map = (error_PAC_map - error_PAC_map_min[k]) / (error_PAC_map_max[k] - error_PAC_map_min[k]) # Normalization
            error_PAC_map[~mask] = - px

            np.save(os.path.join(data_path + "_error",outname + '_' + str(k) + '.npy'), error_PAC_map)

            error_PAC_map_image = error_PAC_map.copy()
            error_PAC_map_image[~mask] = 0

            imageGenerator(image_path + "_error",error_PAC_map_image,mask,od,k,session,outname)
                
            pixels = (pixels - PAC_min) / (PAC_max - PAC_min)

        if ele and ransac!=-1 and k == 4:
            if od:
                pacimage=np.load(os.path.join(data_path,outname.replace('ELE','PAC') +'_0.npy'))
            else:
                pacimage=np.load(os.path.join(data_path,outname.replace('ELE','PAC') + '_0.npy'))
            
            if ransac == 1:
                pixels_rel_enhanced, error_BFS, error_BFS_map = ransac_bfs(pixels, rng, th = th, diskRadius= diskRadius, state = 1)
                
                np.save(os.path.join(error_BFS_path,outname + '_' + str(k) + '.npy'), error_BFS)
                
                error_BFS_map = (error_BFS_map - error_BFS_map_min[k]) / (error_BFS_map_max[k] - error_BFS_map_min[k]) # Normalization

                error_BFS_map[~mask] = - px

                np.save(os.path.join(data_path + "_error",outname + '_' + str(k) + '.npy'), error_BFS_map)

                error_BFS_map_image = error_BFS_map.copy()
                error_BFS_map_image[~mask] = 0

                imageGenerator(image_path + "_error",error_BFS_map_image,mask,od,k,session,outname)
                
            if ransac == 0:
                pixels_rel=quadraticFit(pixels,pacimage,diskRemoval=0)
                pixels_rel_enhanced=quadraticFit(pixels,pacimage,diskRemoval=20)
                # pixels=pixels_rel-pixels_rel_enhanced
            pixels = pixels_rel_enhanced

            pixels = (pixels - ELE_BFS_min[k]) / (ELE_BFS_max[k] - ELE_BFS_min[k]) # Normalization
            pixels = 1 - pixels

        if cd:
            _, error_CD, error_CD_map  = ransac_cd(pixels, rng, th = th * (100/3.5), diskRadius= diskRadius,state = 1)
            if od:
                np.save(os.path.join(error_CD_path,outname + '_' + str(k) + '.npy'), error_CD)
            else:
                np.save(os.path.join(error_CD_path,outname + '_' + str(k) + '.npy'), error_CD)
            
            error_CD_map = (error_CD_map - error_CD_map_min[k]) / (error_CD_map_max[k] - error_CD_map_min[k]) # Normalization
            error_CD_map[~mask] = - px
            
            np.save(os.path.join(data_path + "_error",outname + '_' + str(k) + '.npy'), error_CD_map)

            error_CD_map_image = error_CD_map.copy()
            error_CD_map_image[~mask] = 0

            imageGenerator(image_path + "_error",error_CD_map_image,mask,od,k,session,outname)
            
            pixels = (pixels - CORNEA_DENS_min[k]) / (CORNEA_DENS_max[k] - CORNEA_DENS_min[k]) 

        pixels_img = pixels.copy()
        pixels[~mask]=-px
        
        np.save(os.path.join(data_path,outname + '_' + str(k) + '.npy'), pixels)
        pixels_img[~mask] = 0

        imageGenerator(image_path,pixels_img,mask,od,k,session,outname)
        heatMapGenerator(heatmap_path,pixels_img,od,k,session,outname)
        polarImageGenerator(image_path,polar_image_path,outname,session, k, od)

def polarImageGenerator(fpath,polar_path,outname,session, k, od):

    if od:
        image = cv2.imread(os.path.join(fpath,outname + '_OD_' +session + '_'+ str(k) + '.png'))
    else:
        image = cv2.imread(os.path.join(fpath,outname + '_OS_' +session + '_'+ str(k) + '.png'))
    image  = image.astype(np.float32)/255.0
    value = np.sqrt(((image.shape[0]/2.0)**2.0)+((image.shape[1]/2.0)**2.0))
    polar_image = cv2.linearPolar(image,(image.shape[0]/2, image.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)
    # image = cv2.imwrite(os.path.join(polar_path,outname + '_' + str(k) + '.png'), np.uint16(65535*polar_image),cmap = 'jet')
    if od:
        image = img.imsave(os.path.join(polar_path,outname + '_OD_' +session+'_'+ str(k) + '.png'), polar_image[:,::-1,0].astype(np.float64).T,cmap = 'jet')
    else:
        image = img.imsave(os.path.join(polar_path,outname + '_OS_' + session+ '_'+ str(k) + '.png'), polar_image[:,::-1,0].astype(np.float64).T,cmap = 'jet')
        
    return image

def createIndexFile(patient_idx,df_patients, name, session_index_OS, session_index_OD):
    
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
        
    new_patient = [patient_name,patient_idx,ojo,str(sesion), temp, 'ssl']
    
    return new_patient, session_index_OS, session_index_OD
    
def computeNorm(annFile,data_path):
    
    dataset = pd.read_csv(annFile,header=0, dtype={'Sesion': str})
    #To check the different files
    listFiles=glob.glob(os.path.join(data_path,'1_*.npy'));
    categories=[os.path.basename(fichero).split('_')[1] + '_' + os.path.basename(fichero).split('_')[4] [:-4] for fichero in listFiles]
    mom1=np.zeros(len(categories),)
    mom2=np.zeros(len(categories),)
    cont=np.zeros(len(categories),)
    
    for i in range(len(dataset)):
        
        index = dataset ['Patient'].values[i]
        eye = dataset ['Ojo'].values[i]
        session = dataset ['Sesion'].values[i]
        
        for c in range(len(categories)):
            
            split_cat = categories[c].split('_')
            impath = str(index) + '_' + split_cat[0] + '_' + eye + '_' + str(session) + '_' + split_cat[1] + '.npy'
            im=np.load(data_path+ '/' + impath)
            
            if im is None:
                pdb.set_trace()
            (h,w) = im.shape
            mask = im > -100
            
            #Files
            mom1[c] += im[mask].sum()
            mom2[c] += (im[mask]**2).sum()
            cont[c] += mask.sum()
            
    mean=mom1/cont;
    mom2=mom2/cont;
    
    sigma=np.sqrt(mom2-mean**2);
    return categories,mean,sigma

#Main code
map_cats = ['PAC','CUR','ELE','CORNEA-DENS']

if __name__ == "__main__":
    
    #plt.close('all')
    #plt.cla()
    
    #args = parse_args()
    
    #print('Called with args:')
    #print(args)
    
    #db_path = args.path_prepro
    
    db_path = '../datasets/dataset_ssl_unet'
    
    eConfig = {
        'dir':'RESNET',
        'th':'1.5',
        'r':'45',
        'ransac':'1',
        'pixels':'100'
    }


    #db_path = '../dataset'
    image_path = db_path + '/images'
    data_path = db_path + '/data'

    image_error_path = db_path + '/images_error'
    data_error_path = db_path + '/data_error'

    error_BFS_path = db_path + '/error_BFS' 
    error_PAC_path = db_path + '/error_PAC'
    error_CD_path = db_path + '/error_CD'
    polar_image_path = db_path + '/polar-images' 
    heatmap_path = db_path +  '/heatmaps' 
    df_patients = pd.DataFrame(columns=['Nombre', 'Patient','Ojo', 'Sesion', 'Temp', 'Database'])
    list_path = db_path+'/patient_list.csv'
    
    if not os.path.exists(heatmap_path):
        os.makedirs(heatmap_path)
    
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    if not os.path.exists(image_error_path):
        os.makedirs(image_error_path)
    
    if not os.path.exists(data_error_path):
        os.makedirs(data_error_path)
    
    if not os.path.exists(error_PAC_path):
        os.makedirs(error_PAC_path)
    
    if not os.path.exists(error_BFS_path):
        os.makedirs(error_BFS_path)
    
    if not os.path.exists(error_CD_path):
        os.makedirs(error_CD_path)

    if not os.path.exists(polar_image_path):
        os.makedirs(polar_image_path)
    
    # compute max min
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # ssl database 
    for patient in os.listdir(db_path):
        if patient =='.DS_Store' or patient=='polar-images' or patient=='images' or patient=='heatmaps' or patient=='data' or patient=='error_PAC' or patient =='error_BFS' or patient=='patient_list.csv':
            continue
        patient_path = db_path + '/' + patient
        
        csv_files = glob.glob(patient_path + '/*PAC.CSV')
        
        session_index_OS = 0
        session_index_OD = 0
        
        for f in csv_files:
            for map_cat in map_cats:
                
                if map_cat != 'PAC':
                
                    if re.search(r'_\d{1}b_|_\d{1}_', f):
                        f = re.sub(r'_\d{1}b_|_\d{1}_', '', f)
                    
                csv_name = f.split('/')[-1].replace('PAC',map_cat)
                name = csv_name.replace(".CSV",'')
                split = name.split('_')
                sesion = split[-3]
                if name.find('_OD_') > 0:
                    ojo = 'OD' # OD (right eye)
                else:
                    ojo = 'OS'
                outname = patient + '_' + map_cat + '_' + ojo + '_' + sesion 
                if not os.path.exists(patient_path + '/' + csv_name):
                    print("No ENCONTRADO")
                    continue
                dataset = readCSV(patient_path,csv_name)
                computeMaxMin(dataset, name, outname)
    
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
 
    # compute max min BFS
    np.random.seed(0)
    rng = np.random.default_rng(0)

    for patient in os.listdir(db_path):
        if patient =='.DS_Store' or patient=='polar_images' or patient=='images' or patient=='heatmaps' or patient=='data':
            continue
        patient_path = db_path + '/' + patient
        
        csv_files = glob.glob(patient_path + '/*PAC.CSV')
        
        session_index_OS = 0
        session_index_OD = 0
        
        for f in csv_files:
            for map_cat in map_cats:
                
                if map_cat != 'PAC':
                
                    if re.search(r'_\d{1}b_|_\d{1}_', f):
                        f = re.sub(r'_\d{1}b_|_\d{1}_', '', f)
                    
                csv_name = f.split('/')[-1].replace('PAC',map_cat)
                name = csv_name.replace(".CSV",'')
                split = name.split('_')
                sesion = split[-3]
                if name.find('_OD_') > 0:
                    ojo = 'OD' # OD (right eye)
                else:
                    ojo = 'OS'
                outname = patient + '_' + map_cat + '_' + ojo + '_' + sesion 
                if not os.path.exists(patient_path + '/' + csv_name):
                    print("No ENCONTRADO")
                    continue
                dataset = readCSV(patient_path,csv_name)
                computeMaxMinBFS(data_path, dataset, name, outname, rng, th = float(eConfig['th']), diskRadius = int(eConfig['r']), ransac = int(eConfig['ransac']), px = int(eConfig['pixels']))
    
    for i in np.arange(len(ELE_BFS_max)):
        print('ELE_BFS[{}]'.format(i))
        print('max = ' + str(ELE_BFS_max[i]))
        print('min = ' + str(ELE_BFS_min[i]))

    # generate data
    np.random.seed(0)
    rng = np.random.default_rng(0)
    
    for patient in os.listdir(db_path):
        if patient =='.DS_Store' or patient=='polar_images' or patient=='images' or patient=='heatmaps' or patient=='data':
            continue
        patient_path = db_path + '/' + patient
        
        csv_files = glob.glob(patient_path + '/*PAC.CSV')
        
        session_index_OS = 0
        session_index_OD = 0
        
        for f in csv_files:
            
            for map_cat in map_cats:
                
                if map_cat != 'PAC':
                
                    if re.search(r'_\d{1}b_|_\d{1}_', f):
                        f = re.sub(r'_\d{1}b_|_\d{1}_', '', f)
                    
                csv_name = f.split('/')[-1].replace('PAC',map_cat)
                name = csv_name.replace(".CSV",'')
                split = name.split('_')
                sesion = split[-3]
                if name.find('_OD_') > 0:
                    ojo = 'OD' # OD (right eye)
                else:
                    ojo = 'OS'
                outname = patient + '_' + map_cat + '_' + ojo + '_' + sesion 
                if not os.path.exists(patient_path + '/' + csv_name):
                    print("No ENCONTRADO")
                    continue
                print(outname)
                dataset = readCSV(patient_path,csv_name)
                dataGenerator(data_path, error_BFS_path, error_PAC_path, error_CD_path, image_path, heatmap_path, polar_image_path, dataset,name, outname, patient, rng, th = float(eConfig['th']), diskRadius = int(eConfig['r']), ransac = int(eConfig['ransac']), px = int(eConfig['pixels']))
                
            new_patient, session_index_OS, session_index_OD = createIndexFile(patient,df_patients,name, session_index_OS, session_index_OD)
            df_patients.loc[len(df_patients)] = new_patient
        
    df_patients.to_csv(list_path,index = False)
    cats,mu,sigma = computeNorm(list_path,data_path)
    normalization_data = {'categories': cats, 'mean': mu, 'std': sigma}
    print(normalization_data)
    normalization_file = os.path.join(data_path,'normalization.npz')
    with open(normalization_file, 'wb') as f:
        pickle.dump(normalization_data, f)
            
