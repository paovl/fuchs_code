"""
Created on Wed Fen 12 09:29:16 2025

@author: mariagonzalezgarcia
"""

import torch
import torch.nn.functional as F
import numpy as np
import pdb
from monai.losses import MaskedLoss
from torch.nn import MSELoss as MSELossMonai

class MSELoss(torch.nn.Module):
    def __init__(self, alpha_weighted=0.5):
        super(MSELoss, self).__init__()
        self.alpha_weighted = alpha_weighted

    def forward(self, output, mask, ground_truth, error, normalizer):

        # mask = mask.permute(0, 3, 1, 2)

        # shape = output.shape
        # b = shape[0]

        # output_masked = output[mask]
        # ground_truth_masked = ground_truth[mask]

        # mse_loss = F.mse_loss(output_masked, ground_truth_masked)

        # pdb.set_trace()

        b, c, h, w = output.shape

        output_masked = output * mask	
        ground_truth_masked = ground_truth * mask

        idx_0 = torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 0]
        idx_1 = torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 1]

        number = (mask.sum(dim=(2,3)) > 0).sum(dim=1)[0].item()

        idxs = idx_1.view(b, number)

        output_masked = output_masked[idx_0, idx_1, :, :]
        ground_truth_masked = ground_truth_masked[idx_0, idx_1, :, :]

        output_masked = output_masked.view(b, number, h, w)
        ground_truth_masked = ground_truth_masked.view(b, number, h, w)

        selected_mask = mask[idx_0, idx_1, :, :]
        selected_mask = selected_mask.view(b, number, h, w)

        selected_error = error[idx_0, idx_1, :, :]
        selected_error = selected_error.view(b, number, h, w)

        if len(selected_mask.shape) == 3:
            selected_mask = selected_mask.unsqueeze(1)
            selected_error = selected_error.unsqueeze(1)
            output_masked = output_masked.unsqueeze(1)
            ground_truth_masked = ground_truth_masked.unsqueeze(1)

        mse_loss = F.mse_loss(output_masked, ground_truth_masked, reduction='none')

        selected_mask_error = selected_mask * (1 + self.alpha_weighted * selected_error)

        sum_sample = selected_mask_error.sum(dim=(1,2,3))

        valid_denom = sum_sample != 0

        weighted_mse_loss = mse_loss * selected_mask_error

        map_sum_sample = weighted_mse_loss.sum(dim=(1,2,3))

        map_mean_sample = map_sum_sample / sum_sample

        map_mean_sample[~valid_denom] = 0

        weighted_mse_loss = map_mean_sample.mean()

        # mse_loss = map_mean_sample.mean()

        # output_masked = output * mask
        # ground_truth_masked = ground_truth * mask

        # idx_0 = torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 0]
        # idx_1 = torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 1]
        
        # output_masked = output_masked[idx_0, idx_1, :, :]
        # ground_truth_masked = ground_truth_masked[idx_0, idx_1, :, :]

        # selected_mask = mask[idx_0, idx_1, :, :]

        # if len(selected_mask.shape) == 3:
        #     selected_mask = selected_mask.unsqueeze(1)
        #     output_masked = output_masked.unsqueeze(1)
        #     ground_truth_masked = ground_truth_masked.unsqueeze(1)

        # mse_loss = self.masked_loss(output_masked, ground_truth_masked, selected_mask)
        # pdb.set_trace()

        return weighted_mse_loss, output, ground_truth
