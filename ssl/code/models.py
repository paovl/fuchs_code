#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:28:09 2023

@authors: igonzalez, mariagonzalezgarcia
"""

from torchvision import transforms, utils, models
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pdb
import numpy as np
import matplotlib.pyplot as plt
         
def dimensions_beforePooling(delete_last_layer, model, nMaps):
    if delete_last_layer: 
        model = torch.nn.Sequential(*list(model.children())[:-3]) 
    else: 
        model = torch.nn.Sequential(*list(model.children())[:-2]) 

    input_example = torch.rand(1, nMaps, 352, 112)
    output = model(input_example) # no funciona porque se aplana antes de la fc

    return output.shape[1], output.shape[2], output.shape[3]

class Resnet18Custom (nn.Module):

    def __init__(self, model_ft, branch1, branch2, fc2, fc, fusion):
        super(Resnet18Custom, self).__init__()

        self.model_ft = model_ft
        self.branch1 = branch1
        self.branch2 = branch2
        self.fusion = fusion
        self.sigmoid = nn.Sigmoid()
        self.fc2_extra = fc2
        if fusion == 5:
            self.fc_final = fc
        self.flatten = nn.Flatten()
    
    def forward(self,x, bsizes):

        device  = torch.device("cuda:0")
        features = self.model_ft(x) # back bone output
        batch_size = features.shape[0]

        if self.fusion == 0: # no fusion
          out1 = torch.zeros(batch_size,1).to(device)
          dict = {'features': features, 'bsizes':bsizes}
          out2 = self.branch2(dict) # output branch 2
          out2 = self.flatten(out2)
          out2 = self.fc2_extra(out2)
          return out1

        out1_avgpool = self.branch1(features) # output branch 1
        channels_size = out1_avgpool.shape[1]
        out1 = out1_avgpool.view(batch_size, channels_size) # size adjustment 
        dict = {'features': features, 'bsizes':bsizes}
        out2 = self.branch2(dict) # output branch 2
        out2 = self.flatten(out2)
        out2 = self.fc2_extra(out2)
        
        if self.fusion == 3:
            out = out1 + out2
        elif self.fusion == 5:
            out = torch.cat((out1, out2), dim=1)
            out = self.fc_final(out)
        
        return out
        
class AvgPoolCustom(nn.Module):

    def __init__(self, n_channels, n_rings, n_arcs, mm_per_pixel, degree_per_pixel, rings, arcs):
        super(AvgPoolCustom, self).__init__()
        self.n_rings = n_rings
        self.n_arcs = n_arcs
        self.mm_per_pixel = mm_per_pixel
        self.degree_per_pixel = degree_per_pixel
        self.rings = np.array(rings)
        self.arcs = np.array(arcs)
        # self.ring_mat_mul = self.generate_ring_mat()
        self.angle_mat_mul = self.generate_angle_mat()

    def forward(self, x):

        bsizes = x['bsizes']
        x = x['features']

        bsize_x = bsizes.cpu().numpy()
        mm_per_pixel = ((bsize_x * 0.1)/2) / self.n_rings # revisar

        device = torch.device('cuda:0')

        # Create weighted matrix
        output_mat = np.ndarray((x.shape[0], x.shape[1], len(self.arcs), len(self.rings)))
        output_tensor = torch.tensor(output_mat)
        output_tensor = output_tensor.to(device).float()

        for batch in np.arange(x.shape[0]):
            ring_mat_mul = self.generate_ring_mat(mm_per_pixel[batch])
            for j in np.arange(self.angle_mat_mul.shape[1]):
                for i in np.arange(ring_mat_mul.shape[0]):
                    
                    ring_vect_mul = np.ndarray((1, self.n_rings))
                    ring_vect_mul[0,:] = ring_mat_mul[i,:]
                    angle_vect_mul = np.ndarray((self.n_arcs, 1))
                    angle_vect_mul[:, 0] = self.angle_mat_mul[:,j]
        
                    weights_mat = np.dot(angle_vect_mul, ring_vect_mul)
                    weights_mat = weights_mat / np.sum(weights_mat)
                    weights_tensor = torch.tensor(weights_mat)
                    weights_tensor = weights_tensor.to(device).float()

                    expanded_weights_tensor = weights_tensor.unsqueeze(0)
                    expanded_weights_tensor = expanded_weights_tensor.expand_as(x[batch])

                    weighted_tensor = x[batch] * expanded_weights_tensor

                    weighted_sum = torch.sum(weighted_tensor, dim=(1, 2), keepdim=True)
                    output_tensor[batch, :, j, i] = weighted_sum[:, 0, 0]
            
        return output_tensor
        
        

    def generate_ring_mat(self, mm_per_pixel):
        rings_pixels = np.clip(self.rings / mm_per_pixel, 0, self.n_rings)
        ring_mat_mul = np.ndarray((len(rings_pixels), self.n_rings))
        lower = 0
        for k, ring in enumerate(rings_pixels):
            ring_vect = np.zeros(self.n_rings)
            ring_change = ring - lower
            lower_change = lower -np.floor(lower)
            for i in np.arange(int(np.floor(lower)), self.n_rings):
                upper = np.maximum(0, np.minimum(1-lower_change, ring_change))
                ring_vect[i] = upper
                ring_change = ring_change - upper
                lower_change = 0
            ring_mat_mul[k, :] = ring_vect
            lower = ring
        return ring_mat_mul
    
    def generate_angle_mat(self):
        arcs_pixels = np.clip(self.arcs / self.degree_per_pixel, 0, self.n_arcs)
        angle_mat_mul = np.ndarray((self.n_arcs, len(arcs_pixels)))
        lower = 0
        for k, angle in enumerate(arcs_pixels):
            angle_vect = np.zeros(self.n_arcs)
            angle_change = angle - lower
            lower_change = lower -np.floor(lower)
            for i in np.arange(int(np.floor(lower)), self.n_arcs):
                upper = np.maximum(0, np.minimum(1-lower_change, angle_change))
                angle_vect[i] = upper
                angle_change = angle_change - upper
                lower_change = 0
            angle_mat_mul[:,k] = angle_vect
            lower = angle
        return angle_mat_mul

def replace_layers(model):
    # Replace batch normalization for instance normalization
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module)
            
        if isinstance(module, nn.BatchNorm2d):
            new = nn.InstanceNorm2d(module.num_features,eps=module.eps,momentum=module.momentum, affine=module.affine)
            ## simple module
            try:
                n = int(n)
                model[n] = new
            except:
                setattr(model, n, new)
     
def Resnet18Model(nMaps, rings, arcs, max_radius, max_angle, type, delete_last_layer, fusion, pretrained=True,dropout=0):
    
    model_ft = models.resnet18(weights=None)
    if pretrained:
        model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model_ft = models.resnet18(weights=None)
        model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        aux_conv1 = copy.deepcopy(model_ft.conv1) # Copy the convolutional layer
        new_conv1 = nn.Conv2d(nMaps, aux_conv1.out_channels, aux_conv1.kernel_size, stride=aux_conv1.stride, padding=aux_conv1.padding, dilation=aux_conv1.dilation,
                            groups=aux_conv1.groups, bias=aux_conv1.bias, padding_mode=aux_conv1.padding_mode) # Create a new one with the same characteristics

    aux_conv1 = copy.deepcopy(model_ft.conv1)
    new_conv1 = nn.Conv2d(nMaps, aux_conv1.out_channels, aux_conv1.kernel_size, stride=aux_conv1.stride, padding=aux_conv1.padding, dilation=aux_conv1.dilation,
                          groups=aux_conv1.groups, bias=aux_conv1.bias, padding_mode=aux_conv1.padding_mode) 
    
    #Replicating the first layer
    with torch.no_grad():
        if pretrained:
            for i in range(nMaps):
                rand_data=1e-4*torch.randn_like(new_conv1.weight[:,i,:,:])
                new_conv1.weight[:,i,:,:]=(aux_conv1.weight.mean(dim=1)+rand_data)*aux_conv1.in_channels/nMaps 
            model_ft.conv1 = new_conv1
    
    n_channels, n_arcs, n_rings = dimensions_beforePooling(delete_last_layer, copy.deepcopy(model_ft), nMaps)

    if type == 0: # baseline

        if delete_last_layer == 1:
            model_ft.layer4 = nn.Identity()
        
        model_ft.fc = nn.Identity()
    else: 
        mm_per_pixel = max_radius / n_rings
        degree_per_pixel = max_angle/ n_arcs
        layer4 = copy.deepcopy(model_ft.layer4)
        avg_pool = copy.deepcopy(model_ft.avgpool)

        if type == 1: # no pool

            if delete_last_layer == 1:
                model_ft.fc = nn.Identity()
                model_ft.layer4 = nn.Identity()
                model_ft.avgpool = nn.Identity()
        
            else:
                model_ft.fc = nn.Identity()
                model_ft.avgpool = nn.Identity()

        else: # polar pool

            if delete_last_layer == 1:
                model_ft.fc = nn.Identity()
                model_ft = torch.nn.Sequential(*list(model_ft.children())[:-3])
                branch1 = nn.Sequential(layer4, avg_pool)
            else:
                model_ft.fc = nn.Identity()
                model_ft = torch.nn.Sequential(*list(model_ft.children())[:-2])
                branch1 = nn.Sequential(avg_pool)
            
            if fusion == 5 or fusion == 3:
                fc2 = nn.Linear(n_channels * len(rings) * len(arcs), 512)
                fc = nn.Linear (1024, 512)
                

            branch2 = nn.Sequential(
                AvgPoolCustom(n_channels, n_rings, n_arcs, mm_per_pixel, degree_per_pixel, rings, arcs)
            )
        
            model_ft = Resnet18Custom(model_ft, branch1, branch2, fc2, fc, fusion)

    # Final decision
    

    # Unfreeze first and last layers
    # for param in model_ft.conv1.parameters():
        # param.requires_grad = True

    # for param in model_ft.fc.parameters():
        # param.requires_grad = True

    #Adding some dropout
    if dropout:
        model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
    print('Loading a Resnet18 model')

    #for name, module in model_ft.named_modules():
        #print(name, module)
    return model_ft

# def Resnet18Model(nMaps,pretrained=True,dropout=0, random_seed = 0):

#     torch.manual_seed(random_seed)

#     if pretrained:
#         model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#     else:
#         model_ft = models.resnet18(weights=None)
    
#     aux_conv1 = copy.deepcopy(model_ft.conv1)
#     new_conv1 = nn.Conv2d(nMaps, aux_conv1.out_channels, aux_conv1.kernel_size, stride=aux_conv1.stride, padding=aux_conv1.padding, dilation=aux_conv1.dilation,
#                           groups=aux_conv1.groups, bias=aux_conv1.bias, padding_mode=aux_conv1.padding_mode) 
    
#     #Replicating the first layer
#     with torch.no_grad():
#         if pretrained:
#             for i in range(nMaps):
#                 rand_data=1e-4*torch.randn_like(new_conv1.weight[:,i,:,:])
#                 new_conv1.weight[:,i,:,:]=(aux_conv1.weight.mean(dim=1)+rand_data)*aux_conv1.in_channels/nMaps 
#             model_ft.conv1 = new_conv1
    
    
#     model_ft.fc = nn.Identity()
    
#     #Adding some dropout
#     if dropout:
#         model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
#     print('Loading a Resnet18 model')

#     return model_ft


def Resnet34Model(nMaps,pretrained=True,dropout=0):
    if pretrained:
        model_ft = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        model_ft = models.resnet34(weights=None)
    aux_conv1 = copy.deepcopy(model_ft.conv1)
    new_conv1 = nn.Conv2d(nMaps, aux_conv1.out_channels, aux_conv1.kernel_size, stride=aux_conv1.stride, padding=aux_conv1.padding, dilation=aux_conv1.dilation,
                          groups=aux_conv1.groups, bias=aux_conv1.bias, padding_mode=aux_conv1.padding_mode) 
    
    #Replicating the first layer
    with torch.no_grad():
        if pretrained:
            for i in range(nMaps):
                # j=i%aux_conv1.in_channels
                # new_conv1.weight[:,i,:,:]=aux_conv1.weight[:,j,:,:]#*
                rand_data=1e-4*torch.randn_like(new_conv1.weight[:,i,:,:])
                new_conv1.weight[:,i,:,:]=(aux_conv1.weight.mean(dim=1)+rand_data)*aux_conv1.in_channels/nMaps 
                # new_conv1.weight[:,i,:,:]=torch.median(aux_conv1.weight,dim=1)[0]+1e-4*torch.randn_like(new_conv1.weight[:,i,:,:]) 
            model_ft.conv1 = new_conv1
    
    
    model_ft.fc = nn.Identity()
    
    #Replace bn by instancenorm
    # replace_layers(model_ft)
            

    #Adding some dropout
    if dropout:
        model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
    print('Loading a Resnet34 model')
    return model_ft


def Resnet50Model(nMaps,pretrained=True,dropout=0):
    if pretrained:
        model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model_ft = models.resnet50(weights=None)
    aux_conv1 = copy.deepcopy(model_ft.conv1)
    new_conv1 = nn.Conv2d(nMaps, aux_conv1.out_channels, aux_conv1.kernel_size, stride=aux_conv1.stride, padding=aux_conv1.padding, dilation=aux_conv1.dilation,
                          groups=aux_conv1.groups, bias=aux_conv1.bias, padding_mode=aux_conv1.padding_mode) 
    
    #Replicating the first layer
    with torch.no_grad():
        if pretrained:
            for i in range(nMaps):
                # j=i%aux_conv1.in_channels
                # new_conv1.weight[:,i,:,:]=aux_conv1.weight[:,j,:,:]#*
                rand_data=1e-4*torch.randn_like(new_conv1.weight[:,i,:,:])
                new_conv1.weight[:,i,:,:]=(aux_conv1.weight.mean(dim=1)+rand_data)#*aux_conv1.in_channels/nMaps 
                # new_conv1.weight[:,i,:,:]=torch.median(aux_conv1.weight,dim=1)[0]+1e-4*torch.randn_like(new_conv1.weight[:,i,:,:]) 
            model_ft.conv1 = new_conv1
    
    model_ft.fc = nn.Identity()
    
    #Replace bn by instancenorm
    # replace_layers(model_ft)
            
    #Adding some dropout
    if dropout:
        model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
    print('Loading a Resnet50 model')
    return model_ft

def AlexnetModel(nMaps,pretrained=True,dropout=0):
    if pretrained:
        model_ft = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    else:
        model_ft = models.alexnet(weights=None)
    new_conv1 = copy.deepcopy(model_ft.features[0])
    # new_conv1 = nn.Conv2d(nMaps, aux_conv1.out_channels, aux_conv1.kernel_size, stride=aux_conv1.stride, padding=aux_conv1.padding, dilation=aux_conv1.dilation,
                          # groups=aux_conv1.groups, bias=aux_conv1.bias, padding_mode=aux_conv1.padding_mode) 
    
    #Replicating the first layer
    with torch.no_grad():
        if pretrained:
            for i in range(nMaps):
                # j=i%aux_conv1.in_channels
                # new_conv1.weight[:,i,:,:]=aux_conv1.weight[:,j,:,:]#*
                rand_data=1e-4*torch.randn_like(new_conv1.weight[:,i,:,:])
                new_conv1.weight[:,i,:,:]=(new_conv1.weight.mean(dim=1)+rand_data)#*aux_conv1.in_channels/nMaps 
                # new_conv1.weight[:,i,:,:]=torch.median(aux_conv1.weight,dim=1)[0]+1e-4*torch.randn_like(new_conv1.weight[:,i,:,:]) 
            model_ft.features[0] = new_conv1
    
    num_ftrs = model_ft.classifier[6].in_features
    
    #Replace bn by instancenorm
    # replace_layers(model_ft)
            

    # Final decision
    model_ft.classifier[6] = nn.Identity()
    #Adding some dropout
    # if dropout:
        # model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
    print('Loading a Alexnet model')
    return model_ft

def VGGModel(nMaps,pretrained=True,dropout=0):
    if pretrained:
        model_ft = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:
        model_ft = models.vgg16(weights=None)
    new_conv1 = copy.deepcopy(model_ft.features[0])
    # new_conv1 = nn.Conv2d(nMaps, aux_conv1.out_channels, aux_conv1.kernel_size, stride=aux_conv1.stride, padding=aux_conv1.padding, dilation=aux_conv1.dilation,
                          # groups=aux_conv1.groups, bias=aux_conv1.bias, padding_mode=aux_conv1.padding_mode) 
    
    #Replicating the first layer
    with torch.no_grad():
        if pretrained:
            for i in range(nMaps):
                # j=i%aux_conv1.in_channels
                # new_conv1.weight[:,i,:,:]=aux_conv1.weight[:,j,:,:]#*
                rand_data=1e-4*torch.randn_like(new_conv1.weight[:,i,:,:])
                new_conv1.weight[:,i,:,:]=(new_conv1.weight.mean(dim=1)+rand_data)#*aux_conv1.in_channels/nMaps 
                # new_conv1.weight[:,i,:,:]=torch.median(aux_conv1.weight,dim=1)[0]+1e-4*torch.randn_like(new_conv1.weight[:,i,:,:]) 
            model_ft.features[0] = new_conv1
    
    num_ftrs = model_ft.classifier[6].in_features
    
    #Replace bn by instancenorm
    # replace_layers(model_ft)
            

    # Final decision
    model_ft.classifier[6] = nn.Identity()
    #Adding some dropout
    # if dropout:
        # model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
    print('Loading a VGG model')
    return model_ft

def InceptionModel(nMaps,pretrained=True,dropout=0):
    if pretrained:
        model_ft = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    else:
        model_ft = models.inception_v3(weights=None)
    new_conv1 = copy.deepcopy(model_ft.Conv2d_1a_3x3.conv)
    # new_conv1 = nn.Conv2d(nMaps, aux_conv1.out_channels, aux_conv1.kernel_size, stride=aux_conv1.stride, padding=aux_conv1.padding, dilation=aux_conv1.dilation,
                          # groups=aux_conv1.groups, bias=aux_conv1.bias, padding_mode=aux_conv1.padding_mode) 
    
    #Replicating the first layer
    with torch.no_grad():
        if pretrained:
            for i in range(nMaps):
                # j=i%aux_conv1.in_channels
                # new_conv1.weight[:,i,:,:]=aux_conv1.weight[:,j,:,:]#*
                rand_data=1e-4*torch.randn_like(new_conv1.weight[:,i,:,:])
                new_conv1.weight[:,i,:,:]=(new_conv1.weight.mean(dim=1)+rand_data)#*aux_conv1.in_channels/nMaps 
                # new_conv1.weight[:,i,:,:]=torch.median(aux_conv1.weight,dim=1)[0]+1e-4*torch.randn_like(new_conv1.weight[:,i,:,:]) 
            model_ft.Conv2d_1a_3x3.conv = new_conv1
    
    num_ftrs = model_ft.fc.in_features
    
    #Replace bn by instancenorm
    # replace_layers(model_ft)
            

    # Final decision
    model_ft.fc = nn.Identity()
    #Adding some dropout
    if dropout:
        model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
    print('Loading a Inception model')
    return model_ft



def RegnetY400mModel(nMaps,pretrained=True,dropout=0):
    if pretrained:
        model_ft = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2)
    else:
        model_ft = models.regnet_y_400mf(weights=None)
    aux_conv1 = copy.deepcopy(model_ft.stem[0])
    new_conv1 = nn.Conv2d(nMaps, aux_conv1.out_channels, aux_conv1.kernel_size, stride=aux_conv1.stride, padding=aux_conv1.padding, dilation=aux_conv1.dilation,
                          groups=aux_conv1.groups, bias=aux_conv1.bias, padding_mode=aux_conv1.padding_mode) 
    
    #Replicating the first layer
    with torch.no_grad():
        if pretrained:
            for i in range(nMaps):
                # j=i%aux_conv1.in_channels
                # new_conv1.weight[:,i,:,:]=aux_conv1.weight[:,j,:,:]#*
                rand_data=1e-4*torch.randn_like(new_conv1.weight[:,i,:,:])
                new_conv1.weight[:,i,:,:]=(aux_conv1.weight.mean(dim=1)+rand_data)#*aux_conv1.in_channels/nMaps 
                # new_conv1.weight[:,i,:,:]=torch.median(aux_conv1.weight,dim=1)[0]+1e-4*torch.randn_like(new_conv1.weight[:,i,:,:]) 
            model_ft.stem[0] = new_conv1
    
    num_ftrs = model_ft.fc.in_features
    
    #Replace bn by instancenorm
    # replace_layers(model_ft)
            

    # Final decision
    model_ft.fc = nn.Identity()
    #Adding some dropout
    if dropout:
        model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
    print('Loading a RegNet model')
    return model_ft

def RegnetY800mModel(nMaps,pretrained=True,dropout=0):
    if pretrained:
        model_ft = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V2)
    else:
        model_ft = models.regnet_y_800mf(weights=None)
    aux_conv1 = copy.deepcopy(model_ft.stem[0])
    new_conv1 = nn.Conv2d(nMaps, aux_conv1.out_channels, aux_conv1.kernel_size, stride=aux_conv1.stride, padding=aux_conv1.padding, dilation=aux_conv1.dilation,
                          groups=aux_conv1.groups, bias=aux_conv1.bias, padding_mode=aux_conv1.padding_mode) 
    
    #Replicating the first layer
    with torch.no_grad():
        if pretrained:
            for i in range(nMaps):
                # j=i%aux_conv1.in_channels
                # new_conv1.weight[:,i,:,:]=aux_conv1.weight[:,j,:,:]#*
                rand_data=1e-4*torch.randn_like(new_conv1.weight[:,i,:,:])
                new_conv1.weight[:,i,:,:]=(aux_conv1.weight.mean(dim=1)+rand_data)#*aux_conv1.in_channels/nMaps 
                # new_conv1.weight[:,i,:,:]=torch.median(aux_conv1.weight,dim=1)[0]+1e-4*torch.randn_like(new_conv1.weight[:,i,:,:]) 
            model_ft.stem[0] = new_conv1
    
    num_ftrs = model_ft.fc.in_features
    
    #Replace bn by instancenorm
    # replace_layers(model_ft)
            

    # Final decision
    model_ft.fc = nn.Identity()
    #Adding some dropout
    if dropout:
        model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
    print('Loading a RegNet model')
    return model_ft

def RegnetY3_2gModel(nMaps,pretrained=True,dropout=0):
    if pretrained:
        model_ft = models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2)
    else:
        model_ft = models.regnet_y_3_2gf(weights=None)
    aux_conv1 = copy.deepcopy(model_ft.stem[0])
    new_conv1 = nn.Conv2d(nMaps, aux_conv1.out_channels, aux_conv1.kernel_size, stride=aux_conv1.stride, padding=aux_conv1.padding, dilation=aux_conv1.dilation,
                          groups=aux_conv1.groups, bias=aux_conv1.bias, padding_mode=aux_conv1.padding_mode) 
    
    #Replicating the first layer
    with torch.no_grad():
        if pretrained:
            for i in range(nMaps):
                # j=i%aux_conv1.in_channels
                # new_conv1.weight[:,i,:,:]=aux_conv1.weight[:,j,:,:]#*
                rand_data=1e-4*torch.randn_like(new_conv1.weight[:,i,:,:])
                new_conv1.weight[:,i,:,:]=(aux_conv1.weight.mean(dim=1)+rand_data)#*aux_conv1.in_channels/nMaps 
                # new_conv1.weight[:,i,:,:]=torch.median(aux_conv1.weight,dim=1)[0]+1e-4*torch.randn_like(new_conv1.weight[:,i,:,:]) 
            model_ft.stem[0] = new_conv1
    
    num_ftrs = model_ft.fc.in_features
    
    #Replace bn by instancenorm
    # replace_layers(model_ft)
            

    # Final decision
    model_ft.fc = nn.Identity()
    #Adding some dropout
    if dropout:
        model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
    print('Loading a RegNet model')
    return model_ft

def RegnetY32gModel(nMaps,pretrained=True,dropout=0):
    if pretrained:
        model_ft = models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1)
    else:
        model_ft = models.regnet_y_32gf(weights=None)
    aux_conv1 = copy.deepcopy(model_ft.stem[0])
    new_conv1 = nn.Conv2d(nMaps, aux_conv1.out_channels, aux_conv1.kernel_size, stride=aux_conv1.stride, padding=aux_conv1.padding, dilation=aux_conv1.dilation,
                          groups=aux_conv1.groups, bias=aux_conv1.bias, padding_mode=aux_conv1.padding_mode) 
    
    #Replicating the first layer
    with torch.no_grad():
        if pretrained:
            for i in range(nMaps):
                # j=i%aux_conv1.in_channels
                # new_conv1.weight[:,i,:,:]=aux_conv1.weight[:,j,:,:]#*
                rand_data=1e-4*torch.randn_like(new_conv1.weight[:,i,:,:])
                new_conv1.weight[:,i,:,:]=(aux_conv1.weight.mean(dim=1)+rand_data)#*aux_conv1.in_channels/nMaps 
                # new_conv1.weight[:,i,:,:]=torch.median(aux_conv1.weight,dim=1)[0]+1e-4*torch.randn_like(new_conv1.weight[:,i,:,:]) 
            model_ft.stem[0] = new_conv1
    
    num_ftrs = model_ft.fc.in_features
    
    #Replace bn by instancenorm
    # replace_layers(model_ft)
            

    # Final decision
    model_ft.fc = nn.Identity()
    #Adding some dropout
    if dropout:
        model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
    print('Loading a RegNet model')
    return model_ft


def Mobilev3large(nMaps,pretrained=True,dropout=0):
    if pretrained:
        model_ft = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    else:
        model_ft = models.mobilenet_v3_large(weights=None)
    
    aux_conv1 = copy.deepcopy(model_ft.features[0][0])
    new_conv1 = nn.Conv2d(nMaps, aux_conv1.out_channels, aux_conv1.kernel_size, stride=aux_conv1.stride, padding=aux_conv1.padding, dilation=aux_conv1.dilation,
                          groups=aux_conv1.groups, bias=aux_conv1.bias, padding_mode=aux_conv1.padding_mode) 
    
    #Replicating the first layer
    with torch.no_grad():
        if pretrained:
            for i in range(nMaps):
                # j=i%aux_conv1.in_channels
                # new_conv1.weight[:,i,:,:]=aux_conv1.weight[:,j,:,:]#*
                rand_data=1e-4*torch.randn_like(new_conv1.weight[:,i,:,:])
                new_conv1.weight[:,i,:,:]=(aux_conv1.weight.mean(dim=1)+rand_data)#*aux_conv1.in_channels/nMaps 
                # new_conv1.weight[:,i,:,:]=torch.median(aux_conv1.weight,dim=1)[0]+1e-4*torch.randn_like(new_conv1.weight[:,i,:,:]) 
            model_ft.features[0][0] = new_conv1
    
    num_ftrs = model_ft.classifier[3].in_features
    
    #Replace bn by instancenorm
    # replace_layers(model_ft)
            

    # Final decision
    model_ft.classifier[3] = nn.Identity()
    #Adding some dropout
    if dropout:
        model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=dropout, training=m.training))
    print('Loading a Mobilev3large model')
    return model_ft
