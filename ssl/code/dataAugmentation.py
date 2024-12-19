#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:17:07 2023

@authors: igonzalez
"""
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
import importlib

def mask2box(mask):
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
        
        #image, label= sample['image'],sample['label']
        img1_info, img2_info = sample
        img1 = img1_info['image']
        img2 = img2_info['image']
        # pdb.set_trace()        
        # Cambiamos los ejes
        # numpy 3d volume: H x W x C 
        # torch 3d volume: C x H x W
        h1,w1,c1 = img1.shape
        image1 = img1.transpose((2, 0, 1))
        image1 = torch.from_numpy(image1)
        
        h2,w2,c2 = img2.shape
        image2 = img2.transpose((2,0,1))
        image2 = torch.from_numpy(image2)
        
        return {'image': image1, 'bsize_x':img1_info['bsize_x']}, {'image': image2, 'bsize_x':img2_info['bsize_x']}
    
class Normalize(object):
    """Normalizamos los datos restando la media y dividiendo por las desviaciones típicas.

    Args:
        mean_vec: El vector con las medias. 
        std_vec: el vector con las desviaciones típicas.
    """

    def __init__(self, mean,std):

        assert len(mean)==len(std),'Length of mean and std vectors is not the same'
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        
        img1_info, img2_info = sample
        img1 = img1_info['image']
        img2 = img2_info['image']
        
        normalized_images = {}
        
        for idx, image in enumerate ([img1, img2]):
            mask_bg = image < -1.0
            mask_in = image >= -1.0

            c, h, w = image.shape
            assert c==len(self.mean), 'Length of mean and image is not the same' 
            dtype = image.dtype
            mean = torch.as_tensor(self.mean, dtype=dtype, device=image.device)
            std = torch.as_tensor(self.std, dtype=dtype, device=image.device)
            # print('0',sample["id"],[image[0,...].mean(),image[0,...].std()])
            # print('1',sample["id"],[image[1,...].mean(),image[1,...].std()])
            
            image.sub_(mean[:, None, None]).div_(std[:, None, None])
            image[mask_bg] = -7
            
            idx = idx+1
            
            normalized_images[f'image{idx}'] = image

        return {'image': normalized_images['image1'], 'bsize_x':img1_info['bsize_x']}, {'image': normalized_images['image2'], 'bsize_x':img2_info['bsize_x']}
    
    def denormalize(self, sample):
        """Inverso de la normalización: deshacer la normalización."""
        
        img1_info, img2_info = sample
        img1 = img1_info['image']
        img2 = img2_info['image']
        
        denormalized_images = {}
        
        for idx, image in enumerate([img1, img2]):
            mask_in = image > -7
            mask_bg = ~mask_in
            c, h, w = image.shape
            assert c == len(self.mean), 'Length of mean and image is not the same' 
            dtype = image.dtype
            mean = torch.as_tensor(self.mean, dtype=dtype, device=image.device)
            std = torch.as_tensor(self.std, dtype=dtype, device=image.device)
            
            # Desnormalización: (imagen * std) + media
            image.mul_(std[:, None, None]).add_(mean[:, None, None])
            
            image[mask_bg] = 0
            idx = idx + 1
            denormalized_images[f'image{idx}'] = image
        return {'image': denormalized_images['image1'], 'bsize_x':img1_info['bsize_x']}, {'image': denormalized_images['image2'], 'bsize_x':img2_info['bsize_x']}      

class RandomRotation(object):
    """Rotar la imagen aleatoriamente.

    """
    def __init__(self, angle_range, generator):
        if not isinstance(angle_range, (tuple)):
            angle_range=(-angle_range,angle_range)
        
        self.angle_range = angle_range
        self.generator = generator
        
    def __call__(self, sample):
        

        img1, img2 = sample

        mask_img1_bg = img1 < -1
        mask_img2_bg = img2 < -1
        mask_img1_in = img1 > -1
        mask_img2_in = img2 > -1

        img1[mask_img1_bg] = 0
        img2[mask_img2_bg] = 0
        
        angle=int((self.angle_range[1]-self.angle_range[0])*self.generator.random()+self.angle_range[0])
        
        img_rot1=transform.rotate(img1, angle, mode='constant',preserve_range=True)
        img_rot2=transform.rotate(img2, angle, mode='constant',preserve_range=True)

        for i in range(img_rot1.shape[2]):
            mask_img1 = img_rot1[:,:,i] > 0
            mask_img1 = binary_fill_holes(mask_img1)
            mask_img1 = ~mask_img1
            img_rot1[:,:,i][mask_img1] = -100

            mask_img2 = img_rot2[:,:,i] > 0
            mask_img2 = binary_fill_holes(mask_img2)
            mask_img2 = ~mask_img2
            img_rot2[:,:,i][mask_img2] = -100

        return img_rot1, img_rot2

class RandomTranslation(object):
    
    def __init__(self, max_shift, generator):
        self.max_shift=float(max_shift)
        self.generator = generator

    def __call__(self, sample):

        img1, img2 = sample
        # plt.figure()
        # plt.imshow(image[:,:,0],cmap='gray')
        
        shift_images = {}
        
        for idx, image in enumerate ([img1, img2]):
            [h,w,colors] = image.shape
            translation=image.shape[0:2]*(self.generator.random(2)-0.5); 
            translation=(translation*2.0*self.max_shift).astype(int)
            # print(translation)
            transform = AffineTransform(translation=translation)
            image_shift = warp(image, transform, mode='wrap', preserve_range=True)
            #plt.figure(idx+1)
            #plt.imshow(image_shift[:,:,0],cmap='gray')
            #image_shift[:,:,c] = transform(image[:,:,c])
            #plt.show()
            #pdb.set_trace()
            idx = idx+1
            
            shift_images[f'image{idx}'] = image_shift
                        
        return shift_images['image1'], shift_images['image2']
    
class RandomJitter(object):
    
    def __init__(self, brightness,contrast, generator):
        self.brightness=brightness
        self.contrast=contrast
        self.generator = generator
        
    def __call__(self, sample):

        img1, img2 = sample
        
        jitter_images ={}
        for idx, image in enumerate ([img1, img2]):
            (h,w,channels)=image.shape
            
            for c in range(channels):
                mask_bg = image[:,:,c] < -1
                mask_in = image[:,:,c] > -1
                # plt.figure()
                # plt.imshow(image[:,:,c].clip(0,1),cmap='gray')
                #Sample random variables
                brightness=2*self.brightness*self.generator.random()-self.brightness
                contrast=2*self.contrast*self.generator.random()-self.contrast
                im_jitter=image[:,:,c]*(1+contrast)+brightness
                # plt.figure()
                # plt.imshow(im_jitter.clip(0,1),cmap='gray')
                # plt.show()
                # pdb.set_trace()
                image[:,:,c]=im_jitter
                image[:,:,c] = np.clip(image[:,:,c], 0, None)
                image[:,:,c][mask_bg] = -100
            idx = idx+1
            
            jitter_images[f'image{idx}'] = image
            
                
        return jitter_images['image1'], jitter_images['image2']
    
class centerMinPAC(object):
    
    def __init__(self, mapList):
        self.idxPAC=mapList.index('PAC_0')
        

    def __call__(self, sample):

        img1, img2 = sample
        
        center_images = {}
        
        for idx, image in enumerate ([img1, img2]):
            pac=image[:,:,self.idxPAC]
            # plt.figure();plt.imshow(pac,cmap='gray')
            mask=pac > 0
            # plt.figure();plt.imshow(mask,cmap='gray')
            mask=binary_erosion(mask, footprint=disk(5))
            # plt.figure();plt.imshow(mask,cmap='gray')
            mask=binary_fill_holes(mask)
            # plt.figure();plt.imshow(mask,cmap='gray')
            pac[mask==0]=1
            # plt.figure();plt.imshow(pac,cmap='gray')
            
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
            mask=pac>0;
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
            
            idx = idx+1
            
            center_images[f'image{idx}'] = image_shift
                        
        return center_images['image1'], center_images['image2']

class centerMask(object):
    
    def __init__(self, mapList):
        try:
            self.idxPAC=mapList.index('PAC_0')
        except:
            self.idxPAC=0

    def __call__(self, sample):
        
        img1, img2 = sample
        
        center_mask_images = {}
        
        for idx, image in enumerate ([img1, img2]):
            pac=image[:,:,self.idxPAC]
            # plt.figure();plt.imshow(pac,cmap='gray')
            mask=pac>0;
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
            
            idx = idx+1
            
            center_mask_images[f'image{idx}'] = image_shift
            
                        
        return center_mask_images['image1'], center_mask_images['image2']
    
class cropEye(object):
    
    def __init__(self, mapList,border):
        try:
            self.idxPAC=mapList.index('PAC_0')
        except:
            self.idxPAC=0
        self.border=border

    def __call__(self, sample):
        
        
        img1, img2 = sample
        
        crop_eye_images = {}
        bsize_x = {}
        
        for idx, image in enumerate ([img1, img2]):
            pac=image[:,:,self.idxPAC]
            # plt.figure();plt.imshow(pac.clip(0,1),cmap='gray'); plt.show()
            mask=pac>0
            # plt.figure();plt.imshow(mask,cmap='gray')
            mask=binary_dilation(mask, footprint=disk(self.border))
            # plt.figure();plt.imshow(mask,cmap='gray')
            mask=binary_fill_holes(mask)
            box=mask2simbox(mask)
            image_cropped=image[box[1]:box[3],box[0]:box[2],:]
            # plt.figure();plt.imshow(image_cropped[:,:,0].clip(0,1),cmap='gray');plt.show()
            [h,w,channels]=image_cropped.shape            
            #if h==0 or w==0:
                #pdb.set_trace()
            # print((image.shape,image_cropped.shape))
            idx = idx+1
            crop_eye_images[f'image{idx}'] = image_cropped
            bsize_x[f'image{idx}'] = box[2]-box[0]
            
        return {'image':  crop_eye_images['image1'],'bsize_x':bsize_x['image1']}, {'image':  crop_eye_images['image2'], 'bsize_x':bsize_x['image2']}
    
class centerCrop(object):
    
    def __init__(self, cropSize):
        self.cropSize=cropSize

    def __call__(self, sample):
        
        img1, img2 = sample
        
        crop_center_images = {}
        
        for idx, image in enumerate ([img1, img2]):
            (h,w)=image.shape[0:2]
            center=(w/2,h/2)
            bbox = np.array([center[0]-self.cropSize/2, center[1]-self.cropSize/2, center[0]+self.cropSize/2, center[1]+self.cropSize/2]).astype(int)
            bbox[0:2]=np.maximum(bbox[0:2],0)
            bbox[2]=np.minimum(bbox[2],w)
            bbox[3]=np.minimum(bbox[3],h)
            
            image_cropped=image[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
            # plt.figure();plt.imshow(image_cropped[:,:,0].clip(0,1),cmap='gray');plt.show()
            [h,w,channels]=image_cropped.shape            
            #if h==0 or w==0:
                #pdb.set_trace()
            # print((image.shape,image_cropped.shape))
            
            idx = idx+1
            crop_center_images[f'image{idx}'] = image_cropped
            
            
        return crop_center_images['image1'], crop_center_images['image2']
    
class reSize(object):
    
    def __init__(self, imSize):
        self.imSize=imSize
        

    def __call__(self, sample):
        
        img1_info, img2_info = sample
        img1 = img1_info['image']
        img2 = img2_info['image']
        
        resize_images = {}
        
        for idx, image in enumerate ([img1, img2]):
            mask_bg = image < -1
            mask_in = image > -1
            image[mask_bg] = 0
            # print(image.shape)
            # plt.figure(1)
            # image=image.clip(0,1)
            # plt.imshow(image[:,:,0],cmap='gray')
            [h,w,channels]=image.shape
            image_resized=resize(image,self.imSize,order=1,preserve_range=True) 
            # plt.figure(1);plt.hist(image_resized[image_resized<0].flatten(),bins=100)
            # plt.figure(2);plt.hist(image_resized[image_resized>=0].flatten(),bins=100)
            # plt.show()

            for i in range(image_resized.shape[2]):
                mask = image_resized[:,:,i] > 0
                mask = binary_fill_holes(mask)
                mask= ~mask
                image_resized[:,:,i][mask] = -100
            # image_resized[image_resized<-50]=-100
            # plt.figure(2)
            # plt.imshow(image_resized[:,:,0],cmap='gray')
            # plt.show()
            idx = idx+1
            resize_images[f'image{idx}'] = image_resized
            
        return {'image': resize_images['image1'], 'bsize_x':img1_info['bsize_x']}, {'image': resize_images['image2'], 'bsize_x':img2_info['bsize_x']}      

class PolarCoordinates(object):

    def __call__(self, sample):
        
        img1_info, img2_info = sample
        img1 = img1_info['image']
        img2 = img2_info['image']

        polar_images = {}
        
        for idx, image in enumerate ([img1, img2]):
            mask_bg = image < -1.0
            mask_in = image > -1.0
            image[mask_bg] = 0

            [h,w,channels] = image.shape
            # image  = image.astype(np.float32)/255.0
            # plt.figure(1)
            # image=image.clip(0,1)
            # plt.imshow(image[:,:,0]/image[:,:,0].max(),cmap='gray') 
            value = np.sqrt((image.shape[0]/2)**2 + (image.shape[1]/2)**2)
            polar_image = cv2.warpPolar(image,(-1,-1),(image.shape[0]/2, image.shape[1]/2), image.shape[0]/2, cv2.WARP_FILL_OUTLIERS)
            for i in range(polar_image.shape[2]):
                mask = polar_image[:,:,i] > 0
                mask = binary_fill_holes(mask)
                mask= ~mask
                polar_image[:,:,i][mask] = -100
            if len(polar_image.shape)<3:
                polar_image=polar_image[:,:,np.newaxis]
            #plt.figure(2)
            #plt.imshow(polar_image[:,:,0]/polar_image[:,:,0].max(),cmap='gray')
            #plt.show()
            # pdb.set_trace()
            # src, dsize, center, maxRadius, flags[, dst]	) -> 	dst
            idx = idx+1
            polar_images[f'image{idx}'] = polar_image
            
        return {'image': polar_images['image1'], 'bsize_x':img1_info['bsize_x']}, {'image': polar_images['image2'], 'bsize_x':img2_info['bsize_x']}      

class Deactivate(object):

    def __init__(self, deactivate):
        self.deactivate = deactivate
    def __call__(self, sample):
        img1_info, img2_info = sample
        img1 = img1_info['image']
        img2 = img2_info['image']

        mask_images = {}

        for idx, image in enumerate ([img1, img2]):
            [h, w, channels] = image.shape

            """
            plt.figure(1)
            image_plt  = image[1,:,:].cpu().numpy().astype(np.float32)/255.0
            image_plt = image_plt.clip(0,1)
            plt.imshow(image_plt/image_plt.max(),cmap='gray')
            plt.show()
            """
        
            mask = torch.zeros_like(image)
            mask[:,:,-self.deactivate:] = 1
            mask = mask.bool()
            image[mask] = -7

            """
            plt.figure(2)
            #mask_image_plt  = mask_image[1,:,:].cpu().numpy().astype(np.float32)/255.0
            mask_image_plt = mask_image_plt.clip(0,1)
            plt.imshow(mask_image_plt,cmap='gray')
            plt.show()
            """
            idx = idx+1
            mask_images[f'image{idx}'] = image

        return {'image': mask_images['image1'], 'bsize_x':img1_info['bsize_x']}, {'image': mask_images['image2'], 'bsize_x':img2_info['bsize_x']}      

class cropEyePolar(object):
    
    def __init__(self, radius_px):
        self.radius_px = radius_px

    def __call__(self, sample):
        
        img1, img2 = sample
        
        crop_eye_images = {}
        
        for idx, image in enumerate ([img1, img2]):
            image_cropped = image[:,:,:-self.radius_px]
            idx = idx+1
            crop_eye_images[f'image{idx}'] = image_cropped

        return crop_eye_images['image1'], crop_eye_images['image2']