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

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size = 7, k1 = 0.01, k2 = 0.03, data_range = 1, spatial_dims = 2, device = "cuda:0", alpha_weighted=0.5):
        super(SSIMLoss, self).__init__()
        self.device = device
        self.alpha_weighted = alpha_weighted
        self.kernel_size = window_size
        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range
        self.ssim_loss = SSIMLossMonai(win_size = window_size, k1 = k1, k2 = k2, data_range = data_range, spatial_dims= spatial_dims)
        self.masked_loss = MaskedLoss(self.ssim_loss)

    def forward(self, img1, mask, img2, error, normalizer):

        b, c, h, w = img1.shape

        # img1_new = torch.clamp(normalizer.denormalize(img1), 0, 1)
        # img2_new = torch.clamp(normalizer.denormalize(img2), 0, 1)

        # selected_img1 = img1_new * mask
        # selected_img2 = img2_new * mask

        # selected_img1 = selected_img1.sum(dim=1, keepdim=True)
        # selected_img2 = selected_img2.sum(dim=1, keepdim=True)

        # selected_mask = mask.sum(dim=1, keepdim=True)

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

        selected_error = error[idx_0, idx_1, :, :]
        selected_error = selected_error.view(b, number, h, w)

        if len(selected_mask.shape) == 3:
            selected_mask_ssim = selected_mask.unsqueeze(1)
            selected_error_ssim = selected_error.unsqueeze(1)
            selected_mask = selected_mask.unsqueeze(1)
            selected_error = selected_error.unsqueeze(1)
            selected_img1 = selected_img1.unsqueeze(1)
            selected_img2 = selected_img2.unsqueeze(1)
        else:
            selected_mask_ssim = selected_mask[:, 0, :, :].unsqueeze(1)
            selected_error_ssim = selected_error[:, 0, :, :].unsqueeze(1)
            
        ssim_idx, _ = self.masked_loss(selected_img1, selected_img2, selected_mask_ssim) 

        selected_mask_error = selected_mask * (1 + self.alpha_weighted * selected_error)

        sum_sample = selected_mask_error.sum(dim=(1,2,3))

        valid_denom = sum_sample != 0

        weighted_ssim_idx = ssim_idx * selected_mask_error

        map_sum_sample = weighted_ssim_idx.sum(dim=(1,2,3))

        map_mean_sample = map_sum_sample / sum_sample

        map_mean_sample[~valid_denom] = 0

        weighted_ssim_idx = map_mean_sample.mean()

        weighted_ssim_idx = weighted_ssim_idx * 10

        return weighted_ssim_idx, img1, img2

if __name__ == '__main__':
    ssim_loss = SSIMLoss()
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    mask = torch.rand(1, 3, 256, 256)

    img1 = img1.to('cuda:0')
    img2 = img2.to('cuda:0')
    mask = mask.to('cuda:0')

    print(ssim_loss.forward(img1, mask, img2)[0])