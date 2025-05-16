"""
Created on Wed Fen 12 09:29:16 2025

@author: mariagonzalezgarcia
"""

import torch
import numpy as np
from monai.losses import SSIMLoss as SSIMLossMonai
from monai.losses import MaskedLoss
import matplotlib.pyplot as plt
import pdb

class SSIM_metric(torch.nn.Module):
    def __init__(self, window_size = 7, k1 = 0.01, k2 = 0.03, data_range = 1, spatial_dims = 2, device = "cuda:0", reduction='mean'):
        super(SSIM_metric, self).__init__()
        self.device = device

        self.kernel_size = window_size
        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range
        self.ssim_loss = SSIMLossMonai(win_size = window_size, k1 = k1, k2 = k2, data_range = data_range, spatial_dims= spatial_dims, reduction='none')
        self.reduction = reduction
        self.masked_loss = MaskedLoss(self.ssim_loss)

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

        # img1_new = img1_new.view(-1, 1, h, w)
        # img2_new = img2_new.view(-1, 1, h, w)

        # ssim_idx = self.ssim_loss(img1_new, img2_new)

        # fig, axes = plt.subplots(1, 3, figsize=(15, 10))

        # axes[0].imshow(selected_img1[3, 0, :, :].detach().cpu(), cmap='gray')
        # axes[1].imshow(selected_img1[19, 0, :, :].detach().cpu(), cmap='gray')
        # axes[2].imshow(selected_img1[31, 0, :, :].detach().cpu(), cmap='gray')

        # plt.show()
        # plt.close()

        _, ssim_loss = self.masked_loss(selected_img1, selected_img2, selected_mask)

        ssim_loss = ssim_loss * 10
        
        ssim_metric = (1 - ssim_loss)

        if self.reduction == 'mean':
            ssim_metric = ssim_metric.mean()
         
        return ssim_metric, img1, img2
