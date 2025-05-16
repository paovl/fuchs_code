#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:28:09 2023
@author: igonzalez
"""
import torch
import torch.nn as nn
import numpy as np

class PolarPool(nn.Module):

    def __init__(self, n_channels = 256, n_rings = 7, n_arcs = 22, mm_per_pixel = 1.0, degree_per_pixel = 360/22, rings = [1.0, 3.0, 5.0, 6.0], arcs = [90.0, 180.0, 270.0, 360.0]):
        super(PolarPool, self).__init__()
        self.n_rings = n_rings
        self.n_arcs = n_arcs
        self.mm_per_pixel = mm_per_pixel
        self.degree_per_pixel = degree_per_pixel
        self.rings = np.array(rings)
        self.arcs = np.array(arcs)
        self.angle_mat_mul = self.generate_angle_mat()

    def forward(self, x):
        bsizes = x['bsizes']
        x = x['features']

        bsize_x = bsizes['bsize_x'].cpu().numpy()
        mm_per_pixel = ((bsize_x * 0.1)/2) / self.n_rings # considering 0.1 mm per pixel since the image is 140x140 px and the map length is 14 mm

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
                    if (np.sum(weights_mat) != 0):
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

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):

        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class UNet(nn.Module):
    def __init__(self, n_maps, n_labels):
        super().__init__()

        # ENCODER #
        # in_channels = 3, out_channels=64 for the first block
        self.e1 = encoder_block(n_maps, 16)
        self.e2 = encoder_block(16, 32)
        self.e3 = encoder_block(32, 64)
        self.e4 = encoder_block(64, 128)

        # BOTTLENECK #
        self.b = conv_block(128, 256)

        self.avgp_pool = nn.AdaptiveAvgPool2d((1, 1))

        # LINEAR LAYERS
        self.linear1 = nn.Linear(256, n_labels)

    def forward(self, inputs, bsizes):

        # NOTATION:
        # s -> skip
        # p -> pooling

        # ENCODER #
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # BOTTLENECK #
        b = self.b(p4)

        # LINEAR LAYERS #

        l = self.avgp_pool(b)

        l = l.squeeze()

        outputs = self.linear1(l)

        outputs = outputs.squeeze()

        return outputs

class UNet_PolarPool(nn.Module):
    def __init__(self, n_maps, n_labels):
        super().__init__()

        # ENCODER #
        # in_channels = 3, out_channels=64 for the first block
        self.e1 = encoder_block(n_maps, 16)
        self.e2 = encoder_block(16, 32)
        self.e3 = encoder_block(32, 64)
        self.e4 = encoder_block(64, 128)

        # BOTTLENECK #
        self.b = conv_block(128, 256)

        self.polar_pool = PolarPool()

        # LINEAR LAYERS
        self.linear1 = nn.Linear(4 * 4 * 256, n_labels)

    def forward(self, inputs, bsizes):

        # NOTATION:
        # s -> skip
        # p -> pooling

        # ENCODER #
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # BOTTLENECK #
        b = self.b(p4)

        # LINEAR LAYERS #

        l = self.polar_pool({'features': b, 'bsizes': bsizes})

        # Flatten
        l = l.view(l.size(0), -1)

        outputs = self.linear1(l)

        outputs = outputs.squeeze()

        return outputs

def UNetModel(n_maps, n_labels, model_ssl, freeze=None):
    
    model_ft = UNet(n_maps, n_labels)

    if model_ssl != None:
        model_ft.load_state_dict(model_ssl, strict=False)
    
    if freeze != None:
        for param in model_ft.parameters():
            param.requires_grad = False

        # Descongelar el gradiente de la última capa
        for param in model_ft.linear1.parameters():
            param.requires_grad = True

    print('Loading UNet Model...')

    return model_ft

def UNetModel_PolarPool(n_maps, n_labels, model_ssl, freeze=None):
    
    model_ft = UNet_PolarPool(n_maps, n_labels)

    if model_ssl != None:
        model_ft.load_state_dict(model_ssl, strict=False)
    
    if freeze != None:
        for param in model_ft.parameters():
            param.requires_grad = False

        # Descongelar el gradiente de la última capa
        for param in model_ft.linear1.parameters():
            param.requires_grad = True

    print('Loading UNet Model...')

    return model_ft


    