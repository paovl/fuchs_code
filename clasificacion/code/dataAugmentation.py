#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:17:07 2023

@author: igonzalez
"""
# Imports
import os
import torch
import numpy as np
import pdb
from skimage import io, transform, color, morphology, util
from torchvision.transforms import RandomAffine, ColorJitter
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform, warp
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_erosion, binary_dilation, disk
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.transform import resize
import cv2

def mask2box(mask):
    """

    """
    maski=np.where(mask)
    x_min = int(np.min(maski[1]))
    x_max = int(np.max(maski[1]))
    y_min = int(np.min(maski[0]))
    y_max = int(np.max(maski[0]))

    bbox = np.array([x_min, y_min, x_max, y_max])
    return bbox    

def mask2simbox(mask, border = 10):
    maski = np.where(mask)
    (him,wim) = mask.shape
    x_min = int(np.min(maski[1])) - border
    x_max = int(np.max(maski[1])) + border
    y_min = int(np.min(maski[0])) - border
    y_max = int(np.max(maski[0])) + border
    w=x_max-x_min
    h=y_max-y_min
    center = np.array([int(x_min + w/2),int(y_min + h/2)])
    #Los hago cuadrados
    bsize=np.maximum(w,h)
    bbox = np.array([center[0]-bsize/2, center[1]-bsize/2, center[0]+bsize/2, center[1]+bsize/2]).astype(int)
    #if bbox[0]<0 or bbox[1]<0 or bbox[2]>=wim or bbox[3]>=him:
        #pdb.set_trace()
    bbox[0:2]=np.maximum(bbox[0:2],0)
    bbox[2]=np.minimum(bbox[2],wim)
    bbox[3]=np.minimum(bbox[3],him)
    return bbox     

class ToTensor(object):
    """Convertimos ndarrays de la muestra en tensores."""

    def __call__(self, sample):
        
        # This is the variable the EyeDataset object returns with the function get item
        image_id,image, label= sample['id'],sample['image'],sample['label']
        # pdb.set_trace()        
        # Cambiamos los ejes
        # numpy 3d volume: H x W x C 
        # torch 3d volume: C x H x W
        h,w,c = image.shape
        image = image.transpose((2, 0, 1)) # Reorganize the dimension of the array --> c, h, w
        image = torch.from_numpy(image)
        return {'id':image_id, 'image': image, 'label': label, 'bsize_x':sample['bsize_x'], 'bsize_y':sample['bsize_y']}

class Normalize(object):
    """Normalizamos los datos restando la media y dividiendo por las desviaciones típicas.

    Args:
        mean_vec: Mean vector for each image. 
        std_vec: Standar deviation vector for each image. 
    """

    def __init__(self, mean,std):

        assert len(mean)==len(std),'Length of mean and std vectors is not the same'
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        
        image= sample['image'] # Gets the image
        mask_bg = image<-1 # Obtain the values of the background 
        c, h, w = image.shape
        assert c==len(self.mean), 'Length of mean and image is not the same' 
        dtype = image.dtype
        # Convert the mean and std vector into tensors
        mean = torch.as_tensor(self.mean, dtype=dtype, device=image.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=image.device)
        # print('0',sample["id"],[image[0,...].mean(),image[0,...].std()])
        # print('1',sample["id"],[image[1,...].mean(),image[1,...].std()])

        # Substract the mean and divide with the std
        image.sub_(mean[:, None, None]).div_(std[:, None, None])

        # Why-7? 
        image[mask_bg]=-7
        # print('0',sample["id"],[image[0,...].mean(),image[0,...].std()])
        # print('1',sample["id"],[image[1,...].mean(),image[1,...].std()])

        return {'id':sample["id"],'image': image, 'label' : sample['label'],'bsize_x':sample['bsize_x'], 'bsize_y':sample['bsize_y']}
        
class RandomRotation(object):
    """Rotar la imagen aleatoriamente.
        """
    def __init__(self, angle_range):

        # If its not a tuple, then convert the argument into a tuple 
        if not isinstance(angle_range, (tuple)):
            angle_range=(-angle_range,angle_range)
        
        self.angle_range = angle_range
        
    def __call__(self, sample):
        
        image = sample['image']
        cv2.imwrite("imgs/map_original.png", np.uint16(65535*image[:, :, 1].clip(0,1)))
        angle=int((self.angle_range[1]-self.angle_range[0])*np.random.rand()+self.angle_range[0]) # Random rotation from the min angle range
        img_rot=transform.rotate(image, angle, mode='constant',preserve_range=True)
        cv2.imwrite("imgs/map_rotation.png", np.uint16(65535*img_rot[:, :, 1].clip(0,1)))
        return {'id':sample["id"],'image': img_rot, 'label' : sample['label']}

class RandomTranslation(object):
    
    def __init__(self, max_shift):

        self.max_shift=float(max_shift)

    def __call__(self, sample):
        image = sample['image']
        #plt.figure()
        #plt.imshow(image[:,:,0],cmap='gray')
        [h,w,colors]=image.shape
        translation=image.shape[0:2]*(np.random.rand(2)-0.5)
        translation=(translation*2.0*self.max_shift).astype(int) 
        #print(translation)
        transform = AffineTransform(translation=translation) # Translation 
        image_shift = warp(image, transform, mode='wrap', preserve_range=True)
        # plt.figure()
        # plt.imshow(image_shift[:,:,0],cmap='gray')
            # image_shift[:,:,c] = transform(image[:,:,c])
        # plt.show()
        # pdb.set_trace()
                        
        return {'id':sample["id"],'image': image_shift, 'label' : sample['label']}
    
class RandomJitter(object):
    """Adds a sutile variation of brightness and contrast to the image
    """

    def __init__(self, brightness,contrast):
        self.brightness=brightness
        self.contrast=contrast
        
    def __call__(self, sample):
        image = sample['image']
        (h,w,channels)=image.shape
        # Apply a different variation to each map of the image 
        for c in range(channels):
            # plt.figure()
            # plt.imshow(image[:,:,c].clip(0,1),cmap='gray')
            #Sample random variables
            
            brightness=2*self.brightness*np.random.rand()-self.brightness
            contrast=2*self.contrast*np.random.rand()-self.contrast
            im_jitter=image[:,:,c]*(1+contrast)+brightness
            # plt.figure()
            # plt.imshow(im_jitter.clip(0,1),cmap='gray')
            # plt.show()
            # pdb.set_trace()
            image[:,:,c]=im_jitter
        
        return {'id':sample["id"],'image': image, 'label' : sample['label']}    
    
class centerMinPAC(object):
    def __init__(self, mapList):
        self.idxPAC=mapList.index('PAC_0') # Index of the image channel that has the PAC map
        
    def __call__(self, sample):
        image = sample['image']
        pac=image[:,:,self.idxPAC] # Gets the PAC map
        # plt.figure()
        # plt.imshow(pac,cmap='gray')
        mask=pac>0
        # plt.figure()
        # plt.imshow(mask,cmap='gray')
        mask=binary_erosion(mask, footprint=disk(5)) # Returns and image mask with 5 pixels diameter disk
        # plt.figure()
        # plt.imshow(mask,cmap='gray')
        mask=binary_fill_holes(mask) 
        # plt.figure()
        # plt.imshow(mask,cmap='gray')
        pac[mask==0]=1
        # plt.figure()
        # plt.imshow(pac,cmap='gray')
        
        #Sometimes, there are more than one point with the same density => We low-pass filter the image to be more sure
        px_coords = np.unravel_index(np.argmin(pac, axis=None), pac.shape)
        minVal = pac[px_coords[0],px_coords[1]] 
        if (pac==minVal).sum()>1:
            sigma=0.5
            while 1:
                pacf=gaussian(image[:,:,self.idxPAC], sigma=sigma)
                pacf[mask==0]=1
                px_coords = np.unravel_index(np.argmin(pacf, axis=None), pac.shape)
                minVal = pacf[px_coords[0],px_coords[1]] 
                if (pacf==minVal).sum()==1:
                    break
                sigma=sigma*2
                
        # plt.figure()
        # plt.imshow(image[:,:,0],cmap='gray')
        [h,w,colors]=image.shape
        center=np.array((int(w/2),int(h/2)))
        trans_vector=np.array((center-px_coords[::-1]),dtype=int)
        normTrans=np.linalg.norm(trans_vector)
        transform = AffineTransform(translation=trans_vector)
        image_shift = warp(image, transform, mode='edge', preserve_range=True)
        # if normTrans>20:
        #     print('%f %d'%(normTrans, sample['label'] ))
        #     plt.figure();plt.imshow(pac)
        #     plt.figure();plt.imshow(image_shift[:,:,self.idxPAC])
        #     plt.show()
        pac=image_shift[:,:,self.idxPAC]
        mask=pac>0
        # mask=binary_erosion(mask, footprint=disk(5))
        mask=binary_fill_holes(mask)
        pac[mask==0]=pac[mask==0]+1
        # plt.figure();plt.imshow(pac,cmap='gray')
        px_coords = np.unravel_index(np.argmin(pac, axis=None), pac.shape)
        # plt.show()
        # pdb.set_trace()
        # plt.figure()
        # plt.imshow(image_shift[:,:,0],cmap='gray')
            # image_shift[:,:,c] = transform(image[:,:,c])
        # plt.show()
        # pdb.set_trace()
                        
        return {'id':sample["id"],'image': image_shift, 'label' : sample['label']}

class centerMask(object):
    
    def __init__(self, mapList):
        try:
            self.idxPAC=mapList.index('PAC_0') # Index of the image channel that has the PAC map
        except:
            self.idxPAC=0

    def __call__(self, sample):
        image = sample['image']
        pac=image[:,:,self.idxPAC] # Gets teh PAC map
        # plt.figure();plt.imshow(pac,cmap='gray')
        mask=pac>0
        # plt.figure();plt.imshow(mask,cmap='gray')
        # mask=binary_erosion(mask, footprint=disk(2))
        # plt.figure();plt.imshow(mask,cmap='gray')
        # mask=binary_fill_holes(mask)
        mask=mask.astype(dtype=np.int32)
        regprops = regionprops(mask)
        y0, x0 = regprops[0].centroid
        centroid=np.array((x0,y0),dtype=int)
        [h,w,colors]=image.shape
        center=np.array((int(w/2),int(h/2)))
        transform = AffineTransform(translation=(center-centroid))
        normTrans=np.linalg.norm((center-centroid))
        image_shift = warp(image, transform, mode='edge', preserve_range=True)
        
        # if normTrans>5:
        #     print('%f %d'%(normTrans, sample['label'] ))
        #     plt.figure();plt.imshow(pac)
        #     plt.figure();plt.imshow(image_shift[:,:,self.idxPAC])
        #     plt.show()
                        
        return {'id':sample["id"],'image': image_shift, 'label' : sample['label']}
    
class cropEye(object):
    """ This function gets the image with only the eye in a circle 
    """
    def __init__(self, mapList,border):
        try:
            self.idxPAC=mapList.index('PAC_0') # Index of the image channel that has the PAC map
        except:
            self.idxPAC=0
        self.border=border

    def __call__(self, sample):
        
        image = sample['image']
        pac=image[:,:,self.idxPAC]
        # plt.figure();plt.imshow(pac.clip(0,1),cmap='gray')
        mask=pac>0
        # plt.figure();plt.imshow(mask,cmap='gray')
        mask=binary_dilation(mask, footprint=disk(self.border))
        # plt.figure();plt.imshow(mask,cmap='gray')
        # mask=binary_fill_holes(mask)
        box=mask2simbox(mask)
        image_cropped=image[box[1]:box[3],box[0]:box[2],:]
        # plt.figure();plt.imshow(image_cropped[:,:,0].clip(0,1),cmap='gray');plt.show()
        [h,w,channels]=image_cropped.shape            
        if h==0 or w==0:
            pdb.set_trace()
        # print((image.shape,image_cropped.shape))
        return {'id':sample["id"],'image': image_cropped, 'label' : sample['label'], 'bsize_x':box[2]-box[0], 'bsize_y':box[3]-box[1]}

class centerCrop(object):
    
    def __init__(self, cropSize):
        self.cropSize=cropSize

    def __call__(self, sample):
        
        image = sample['image']
        (h,w)=image.shape[0:2]
        center=(w/2,h/2)
        bbox = np.array([center[0]-self.cropSize/2, center[1]-self.cropSize/2, center[0]+self.cropSize/2, center[1]+self.cropSize/2]).astype(int)
        bbox[0:2]=np.maximum(bbox[0:2],0)
        bbox[2]=np.minimum(bbox[2],w)
        bbox[3]=np.minimum(bbox[3],h)
        
        image_cropped=image[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        # plt.figure();plt.imshow(image_cropped[:,:,0].clip(0,1),cmap='gray');plt.show()
        [h,w,channels]=image_cropped.shape            
        if h==0 or w==0:
            pdb.set_trace()
        # print((image.shape,image_cropped.shape))
        return {'id':sample["id"],'image': image_cropped, 'label' : sample['label']}
    
class reSize(object):
    
    def __init__(self, imSize):
        self.imSize=imSize       

    def __call__(self, sample):
        image = sample['image']
        # print(image.shape)
        # plt.figure(1)
        # image=image.clip(0,1)
        # plt.imshow(image[:,:,0],cmap='gray')
        [h,w,channels]=image.shape
        image_resized=resize(image,self.imSize,order=1,preserve_range=True) 
        # plt.figure(1);plt.hist(image_resized[image_resized<0].flatten(),bins=100)
        # plt.figure(2);plt.hist(image_resized[image_resized>=0].flatten(),bins=100)
        # plt.show()
        # image_resized[image_resized<-50]=-100
        # plt.figure(2)
        # plt.imshow(image_resized[:,:,0],cmap='gray')
        # plt.show()
        return {'id':sample["id"],'image': image_resized, 'label' : sample['label'], 'bsize_x':sample['bsize_x'], 'bsize_y':sample['bsize_y']}

class PolarCoordinates(object):
    def __init__(self, max_radius, max_degrees):
        self.max_radius = max_radius
        self.max_degrees = max_degrees
    def __call__(self, sample):
        image = sample['image']
        [h,w,channels]=image.shape
        #image  = image.astype(np.float32)/255.0
        #plt.figure(1)
        #image=image.clip(0,1)
        #plt.imshow(image[:,:,0]/image[:,:,0].max(),cmap='gray')  
        polar_image = cv2.warpPolar(image,(-1,-1),(image.shape[0]/2, image.shape[1]/2), self.max_radius, cv2.WARP_FILL_OUTLIERS)
        idx_angle = self.max_degrees * polar_image.shape[0] / 360
        polar_image = polar_image[0:int(idx_angle), :]
        if len(polar_image.shape)<3:
            polar_image=polar_image[:,:,np.newaxis]
        """
        polar_image_plt = polar_image.astype(np.float32)/255.0
        polar_image_plt=polar_image_plt.clip(0,1)
        plt.figure(2)
        plt.imshow(polar_image_plt[:,:,0]/polar_image_plt[:,:,0].max(),cmap='gray')
        """
        #plt.show()
        #cv2.imwrite("polar_images/polar_"+ str(sample['id'])+ ".png", np.uint16(65535*(polar_image_plt[:,:,0]/polar_image_plt[:,:,0].max())))
        #pdb.set_trace()
        # src, dsize, center, maxRadius, flags[, dst]	) -> 	dst
        return {'id':sample["id"],'image': polar_image , 'label' : sample['label'], 'bsize_x':sample['bsize_x'], 'bsize_y':sample['bsize_y']}
    
# class quadraticFit(image):
#     ordr = 2  # order of polynomial
#     mask=image>0;
#     mask=binary_fill_holes(mask)
#     px_coords = np.unravel_index(np.where(mask, axis=None), mask.shape)
    
#     points=image[]
#     x, y, z = points
#     x, y = x - x[0], y - y[0]  # this improves accuracy

# # make Matrix:
# G = poly_matrix(x, y, ordr)
# # Solve for np.dot(G, m) = z:
# m = np.linalg.lstsq(G, z)[0]
