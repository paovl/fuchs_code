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
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, mask, ground_truth, normalizer):

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

        if len(selected_mask.shape) == 3:
            selected_mask = selected_mask.unsqueeze(1)
            output_masked = output_masked.unsqueeze(1)
            ground_truth_masked = ground_truth_masked.unsqueeze(1)

        mse_loss = F.mse_loss(output_masked, ground_truth_masked, reduction='none')

        sum_sample = mask.sum(dim=(1,2,3))

        map_sum_sample = mse_loss.sum(dim=(1,2,3))

        map_mean_sample = map_sum_sample / sum_sample

        mse_loss = map_mean_sample.mean()

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

        return mse_loss, output, ground_truth
