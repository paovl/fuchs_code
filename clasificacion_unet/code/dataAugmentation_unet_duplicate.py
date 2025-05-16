#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:17:07 2023

@authors: igonzalez
"""
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform, warp
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_erosion, binary_dilation, disk
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.transform import resize
import cv2
from skimage import transform
import pdb


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
        img_info = sample
        img = img_info['image']

        if img_info.get('label', None) is not None:
            label = img_info['label']
        else: 
            label = img.copy()

        if img_info.get('mask', None) is not None:
            mask = img_info['mask']
        else:
            mask = np.ones_like(img, dtype=bool)
        # pdb.set_trace()        
        # Cambiamos los ejes
        # numpy 3d volume: H x W x C 
        # torch 3d volume: C x H x W
        image = img.transpose((0, 3, 1, 2))
        label = label.transpose((0, 3, 1, 2))

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return {'image': image, 'label': label, 'mask': mask}

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
        
        img_info = sample
        image = img_info['image']

        if img_info.get('label', None) is not None:
            label = img_info['label']
        else: 
            label = image.copy()

        if img_info.get('mask', None) is not None:
            mask = img_info['mask']
        else:
            mask = np.ones_like(image, dtype=bool)

        mask_bg = image < -1
        mask_bg_l = label < -1

        image[mask_bg] = 0
        label[mask_bg_l] = 0

        b, c, h, w = image.shape
        b_l, c_l, h_l, w_l = label.shape

        assert c==len(self.mean), 'Length of mean and image is not the same' 
        assert c_l==len(self.mean), 'Length of mean and label is not the same'

        dtype = image.dtype
        dtype_l = label.dtype

        mean = torch.as_tensor(self.mean, dtype=dtype, device=image.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=image.device)
        mean_l = torch.as_tensor(self.mean, dtype=dtype_l, device=label.device)
        std_l = torch.as_tensor(self.std, dtype=dtype_l, device=label.device)

        image.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        # image[mask_bg] = -7
        # image[mask_bg] = min(mean) - (max(std) * 20)
        label.sub_(mean_l[None, :, None, None]).div_(std_l[None, :, None, None])
        # label[mask_bg_l] = -7
        # label[mask_bg_l] = min(mean) - (max(std) * 20)

        normalized_image = image
        normalized_label = label
    
        return {'image': normalized_image, 'label': normalized_label, 'mask': mask}
    
    def denormalize(self, sample):
        """Inverso de la normalización: deshacer la normalización."""
        image = sample

        dtype = image.dtype
        
        mean = torch.as_tensor(self.mean, dtype=dtype, device=image.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=image.device)
        
        # mask_in = image > -7
        # mask_in = image >  min(mean) - (max(std) * 20)
        # mask_bg = ~mask_in

        if len(image.shape) == 3:
            c, h, w = image.shape
            assert c == len(self.mean), 'Length of mean and image is not the same' 
            dtype = image.dtype

            mean = torch.as_tensor(self.mean, dtype=dtype, device=image.device)
            std = torch.as_tensor(self.std, dtype=dtype, device=image.device)
            
            # Desnormalización: (imagen * std) + media
            image.mul_(std[:, None, None]).add_(mean[:, None, None])
            
            # image[mask_bg] = 0

            denormalized_image = image

        elif len(image.shape) == 4:

            b, c, h, w = image.shape
            assert c == len(self.mean), 'Length of mean and image is not the same' 

            dtype = image.dtype
            device = image.device

            mean = torch.as_tensor(self.mean, dtype=dtype, device=device).view(1, c, 1, 1)
            std = torch.as_tensor(self.std, dtype=dtype, device=device).view(1, c, 1, 1)

            # Desnormalización por lotes
            image = image * std + mean
            
            # Restaurar valores de fondo a 0
            # image[mask_bg] = 0

            denormalized_image = image
        
        return denormalized_image

class RandomRotation(object):
    """Rotar la imagen aleatoriamente.

    """
    def __init__(self, angle_range, generator):
        if not isinstance(angle_range, (tuple)):
            angle_range=(-angle_range,angle_range)
        
        self.angle_range = angle_range
        self.generator = generator
        
    def __call__(self, sample):

        img = sample['image']

        mask_img_bg = img < -1

        img[mask_img_bg] = 0
        
        angle = int((self.angle_range[1]-self.angle_range[0])*self.generator.random()+self.angle_range[0])
        
        img_rot = transform.rotate(img, angle, mode='constant',preserve_range=True)

        for i in range(img_rot.shape[2]):
            mask_img = img_rot[:,:,i] > 0
            mask_img = binary_fill_holes(mask_img)
            mask_img = ~mask_img
            img_rot[:,:,i][mask_img] = -100

        return {'image': img_rot}

class RandomTranslation(object):
    
    def __init__(self, max_shift, generator):
        self.max_shift=float(max_shift)
        self.generator = generator

    def __call__(self, sample):

        image = sample['image']
        # plt.figure()
        # plt.imshow(image[:,:,0],cmap='gray')
        
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
            
        shift_image = image_shift
                        
        return {'image': shift_image}
    
class RandomJitter(object):
    
    def __init__(self, brightness,contrast, generator):
        self.brightness=brightness
        self.contrast=contrast
        self.generator = generator
        
    def __call__(self, sample):

        image = sample['image']
        
        (h,w,channels)=image.shape
            
        for c in range(channels):
            mask_bg = image[:,:,c] < -1
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

        jitter_image = image
                 
        return {'image': jitter_image}
    
class centerMinPAC(object):
    
    def __init__(self, mapList):
        self.idxPAC=mapList.index('PAC_0')
        

    def __call__(self, sample):

        image = sample['image']

        pac = image[:,:,self.idxPAC]
        # plt.figure();plt.imshow(pac,cmap='gray')
        mask = pac > 0
        # plt.figure();plt.imshow(mask,cmap='gray')
        mask = binary_erosion(mask, footprint=disk(5))
        # plt.figure();plt.imshow(mask,cmap='gray')
        mask = binary_fill_holes(mask)
        # plt.figure();plt.imshow(mask,cmap='gray')
        pac[mask == 0] = 1
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
        [h,w,colors] = image.shape
        center = np.array((int(w/2),int(h/2)))
        trans_vector = np.array((center-px_coords[::-1]),dtype=int)
        normTrans=np.linalg.norm(trans_vector)
        transform = AffineTransform(translation=trans_vector)
        image_shift = warp(image, transform, mode='edge', preserve_range=True)
        # if normTrans>20:
        #     print('%f %d'%(normTrans, sample['label'] ))
        #     plt.figure();plt.imshow(pac)
        #     plt.figure();plt.imshow(image_shift[:,:,self.idxPAC])
        #     plt.show()
        pac = image_shift[:,:,self.idxPAC]
        mask = pac > 0
        # mask=binary_erosion(mask, footprint=disk(5))
        mask=binary_fill_holes(mask)
        pac[mask == 0] = pac[mask == 0] + 1
        # plt.figure();plt.imshow(pac,cmap='gray')
        px_coords = np.unravel_index(np.argmin(pac, axis=None), pac.shape)
        # plt.show()
        # pdb.set_trace()
        # plt.figure()
        # plt.imshow(image_shift[:,:,0],cmap='gray')
            # image_shift[:,:,c] = transform(image[:,:,c])
        # plt.show()
        # pdb.set_trace()
            
        center_image = image_shift
                        
        return {'image': center_image}

class centerMask(object):
    
    def __init__(self, mapList):
        try:
            self.idxPAC=mapList.index('PAC_0')
        except:
            self.idxPAC=0

    def __call__(self, sample):
        
        image = sample['image']
        
        pac = image[:,:,self.idxPAC]
        # plt.figure();plt.imshow(pac,cmap='gray')
        mask = pac > 0
        # plt.figure();plt.imshow(mask,cmap='gray')
        # mask=binary_erosion(mask, footprint=disk(2))
        # plt.figure();plt.imshow(mask,cmap='gray')
        # mask=binary_fill_holes(mask)
        mask = mask.astype(dtype = np.int32)
        regprops = regionprops(mask)
        y0, x0 = regprops[0].centroid
        centroid=np.array((x0,y0),dtype=int)
        [h,w,colors] = image.shape
        center = np.array((int(w/2),int(h/2)))
        transform = AffineTransform(translation = (center-centroid))
        normTrans = np.linalg.norm((center-centroid))
        image_shift = warp(image, transform, mode = 'edge', preserve_range = True)
        
        # if normTrans>5:
        #     print('%f %d'%(normTrans, sample['label'] ))
        #     plt.figure();plt.imshow(pac)
        #     plt.figure();plt.imshow(image_shift[:,:,self.idxPAC])
        #     plt.show()

        
        center_mask_image = image_shift
                        
        return {'image': center_mask_image}
    
class cropEye(object):
    
    def __init__(self, mapList,border):
        try:
            self.idxPAC=mapList.index('PAC_0')
        except:
            self.idxPAC=0
        self.border=border

    def __call__(self, sample):

        
        image = sample['image']

        if self.border < 0:
            return {'image': image }

    
        pac = image[:,:,self.idxPAC]
        # plt.figure();plt.imshow(pac.clip(0,1),cmap='gray'); plt.show()
        mask = pac>0
        # plt.figure();plt.imshow(mask,cmap='gray')
        mask = binary_dilation(mask, footprint=disk(self.border))
        # plt.figure();plt.imshow(mask,cmap='gray')
        mask = binary_fill_holes(mask)
        box = mask2simbox(mask)
        image_cropped = image[box[1]:box[3],box[0]:box[2],:]
        # plt.figure();plt.imshow(image_cropped[:,:,0].clip(0,1),cmap='gray');plt.show()
        [h,w,channels] = image_cropped.shape            
        #if h==0 or w==0:
            #pdb.set_trace()
        # print((image.shape,image_cropped.shape))
        crop_eye_image = image_cropped
        bsize = box[2]-box[0]
        
        return {'image':  crop_eye_image}

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
        #if h==0 or w==0:
            #pdb.set_trace()
        # print((image.shape,image_cropped.shape))
        
        crop_center_image = image_cropped
        
        return crop_center_image
    
class reSize(object):
    
    def __init__(self, imSize):
        self.imSize = imSize 

    def __call__(self, sample):
        
        img_info = sample
        image = img_info['image']
        
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

        resize_image = image_resized
        
        return {'image': resize_image}
    
class PolarCoordinates(object):

    def __call__(self, sample):
        
        img_info = sample
        image = img_info['image']

        mask_bg = image < -1.0
        mask_in = image > -1.0
        image[mask_bg] = 0

        if len(image.shape) == 2:
            image = image.reshape((image.shape[0], image.shape[1], 1))

        [h,w,channels] = image.shape
        # image  = image.astype(np.float32)/255.0
        # plt.figure(1)
        # image=image.clip(0,1)
        # plt.imshow(image[:,:,0]/image[:,:,0].max(),cmap='gray') 
        value = np.sqrt((image.shape[0]/2)**2 + (image.shape[1]/2)**2)
        polar_image = cv2.warpPolar(image,(-1,-1),(image.shape[0]/2, image.shape[1]/2), image.shape[0]/2, cv2.WARP_FILL_OUTLIERS)
        if len(polar_image.shape) == 2:
            polar_image = polar_image.reshape((polar_image.shape[0], polar_image.shape[1], 1))
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
        
        return {'image': polar_image}

class Deactivate(object):


    def __init__(self, deactivate):
        self.deactivate = deactivate
    def __call__(self, sample):
        img_info, img_info = sample
        image = img_info['image']

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

        mask_image = image

        return {'image': mask_image}

class cropEyePolar(object):
    
    def __init__(self, radius_px):
        self.radius_px = radius_px

    def __call__(self, sample):
        
        image = sample
        
        image_cropped = image[:,:,:-self.radius_px]
        crop_eye_image = image_cropped

        return crop_eye_image

class PatchHide_duplicate(object):
    def __init__(self, generator, dilation = 0):
        self.generator = generator
        self.dilation = dilation

    def __call__(self, sample):
        img = sample
        image = img['image'] # dimensions --> 352 (angles) x 112 (radii) x 3 (maps)

        h, w, c = image.shape  # 352 x 112 x 3
        mask = np.zeros((c, image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32) # Inicializar máscara con 1s

        image = np.repeat(image[np.newaxis, :, :, :], c, axis=0) # 3 x 352 x 112 x 3
        
        for j in range(c):
            mask[j, :, :, j] = 1
        
        mask_before = image > -100 # Máscara de valores que no son fondo
        image_before = image.copy()
        image_before[~mask_before] = 0

        mask = mask.astype(bool)
        masked_image = image.copy()
        masked_image[mask] = -100

        mask = mask_before & mask

        if self.dilation > 0:
            mask_dilation = mask.copy()
            for i in range(c):
                mask_dilation[i, :, :, i] = binary_dilation(mask_dilation[i, :, :, i], footprint=disk(self.dilation))
                mask_dilation[i, :, :, i] = binary_fill_holes(mask_dilation[i, :, :, i])
            mask_dilation = mask_dilation.astype(bool)
            mask = mask_dilation  
        
        # fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # for i in range(3):
        #     axes[0, i].imshow(mask[2, :, :, i], cmap='gray')
        #     axes[0, i].set_title(f'Masked map {i}')

        #     axes[1, i].imshow(mask_dilation[2, :, :, i], cmap='gray')
        #     axes[1, i].set_title(f'Dilation masked map {i}')

        # plt.show()

        mask_after = image > - 100
        image_after = image.copy()
        image_after[~mask_after] = 0

        return {'image': masked_image, 'label': image, 'mask': mask}