#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 23:13:46 2024

@author: mariagonzalezgarcia
"""

import torch
import torch.nn.functional as F
import pdb
import numpy as np
import matplotlib.pyplot as plt


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5, alpha=0.5, th_weights = 0, a_weights = 1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha  = alpha
        self.th_weights = th_weights
        self.a_weights = a_weights

    def forward(self, output, error_PAC, error_BFS, epoch, nbatch, valset = False):
        error = np.zeros((error_PAC.shape[0], 2)) # vector with PAC error by pairs 
        error[:,0] = error_PAC[:,0]
        error[:,1] = error_PAC[:,1]

        weights_loss = np.maximum(error[:, 0], error[:, 1])
        
        device = output.device
        n,d = output.shape
        labels = torch.arange(0,int(n/2), dtype=torch.float)
        labels = torch.tile(labels,(2,))
        plabels = F.pdist(labels.unsqueeze(1), p=0)
        pdist = F.pdist(output, p=2)
        idx = np.arange(0, n)

        weights_loss_norm = weights_loss / np.sum(weights_loss)

        binary_weights = np.where(weights_loss < self.th_weights, 0, 1)
        sigmoid_weights = 1 / (1 + np.exp(-self.a_weights*(weights_loss - self.th_weights)))
       
        idx_weights_binary = np.where(binary_weights == 1)[0]
        idx = np.where(plabels==0)[0]
        intersected_idx = np.intersect1d(idx_weights_binary, idx)
        
        # ppost = sum(plabels==0) / len(plabels)
        # idx = np.where(plabels==0)[0]
        # ppost = torch.tensor(sum(weights_loss_norm[idx]))
        # ppost = sum(plabels[intersected_idx] == 0) / len(plabels[idx_weights_binary])
        ppost = torch.tensor(sum(sigmoid_weights[idx]) / sum(sigmoid_weights))

        weights_loss_norm = torch.from_numpy(weights_loss_norm).to(device)
        weights_loss = torch.from_numpy(weights_loss).to(device)
        binary_weights = torch.from_numpy(binary_weights).to(device)
        sigmoid_weights = torch.from_numpy(sigmoid_weights).to(device)
        ppost = ppost.to(device)  # Ensure all tensors are on the same device
        plabels = plabels.to(device)
        pdist = pdist.to(device)

        # loss_contrastive = torch.mean(((1-ppost)**self.alpha)*(1 - plabels) * torch.pow(pdist, 2) +
        #                                (ppost**self.alpha) * plabels * torch.pow(torch.clamp(self.margin - pdist, min=0.0), 2))*2

        # loss_contrastive = torch.mean(weights_loss_norm * (((1-ppost)**self.alpha)*(1 - plabels) * torch.pow(pdist, 2) +
        #                             (ppost**self.alpha) * plabels * torch.pow(torch.clamp(self.margin - pdist, min=0.0), 2)))*2
        
        # loss_contrastive = torch.mean(binary_weights * (((1-ppost)**self.alpha)*(1 - plabels) * torch.pow(pdist, 2) +
        #                             (ppost**self.alpha) * plabels * torch.pow(torch.clamp(self.margin - pdist, min=0.0), 2)))*2

        loss_contrastive = torch.mean(sigmoid_weights * (((1-ppost)**self.alpha)*(1 - plabels) * torch.pow(pdist, 2) +
                                    (ppost**self.alpha) * plabels * torch.pow(torch.clamp(self.margin - pdist, min=0.0), 2)))*2

        pred_sim = torch.exp(-pdist)
        
        return loss_contrastive, 1-plabels, pred_sim, weights_loss, weights_loss_norm, binary_weights, sigmoid_weights, len(weights_loss), idx

