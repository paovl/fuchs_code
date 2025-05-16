"""
Created on Wed Fen 12 09:29:16 2025

@author: mariagonzalezgarcia
"""

import torch
import torch.nn.functional as F
import numpy as np
from monai.losses import LocalNormalizedCrossCorrelationLoss
from monai.losses import MaskedLoss
import pdb

class LocalNCC_metric(torch.nn.Module):
    def __init__(self, spatial_dims = 2, kernel_size = 3, kernel_type = "rectangular", reduction = 'mean'):
        super(LocalNCC_metric, self).__init__()
        self.local_ncc_loss = LocalNormalizedCrossCorrelationLoss(spatial_dims= spatial_dims, kernel_size = kernel_size, kernel_type = kernel_type, reduction = 'none')
        self.masked_loss = MaskedLoss(self.local_ncc_loss)
        self.reduction = reduction

    def forward(self, img1, mask, img2, normalizer):

        b, c, h, w = img1.shape

        img1_new = torch.clamp(normalizer.denormalize(img1), 0, 1)
        img2_new = torch.clamp(normalizer.denormalize(img2), 0, 1)

        selected_img1 = img1_new * mask
        selected_img2 = img2_new * mask

        idx_0 = torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 0]
        idx_1 = torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 1]

        number = (mask.sum(dim=(2,3)) > 0).sum(dim=1)[0].item()

        idxs = idx_1.view(b, number)

        selected_img1 = selected_img1[idx_0, idx_1, :, :]
        selected_img2 = selected_img2[idx_0, idx_1, :, :]

        selected_img1 = selected_img1.view(b, number, h, w)
        selected_img2 = selected_img2.view(b, number, h, w)

        selected_mask = mask[idx_0, idx_1, :, :]

        selected_mask = selected_mask.view(b, number, h, w)

        if len(selected_mask.shape) == 3:
            selected_mask = selected_mask.unsqueeze(1)
            selected_img1 = selected_img1.unsqueeze(1)
            selected_img2 = selected_img2.unsqueeze(1)
        else:
            selected_mask = selected_mask[:, 0, :, :].unsqueeze(1)

        local_ncc = self.masked_loss(selected_img1, selected_img2, selected_mask) # Devuelve negada

        local_ncc = -local_ncc

        sum_sample = mask.sum(dim=(1,2,3))

        map_sum_sample = local_ncc.sum(dim=(1,2,3))

        local_ncc = map_sum_sample / sum_sample

        if self.reduction == 'mean':
            local_ncc_metric = local_ncc.mean()
        else:
            local_ncc_metric = local_ncc
        
        return local_ncc_metric, img1, img2
