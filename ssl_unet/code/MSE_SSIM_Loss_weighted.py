"""
Created on Wed Fen 12 09:29:16 2025

@author: mariagonzalezgarcia
"""

import torch
import torch.nn.functional as F
import numpy as np
from MSELoss_weighted import MSELoss
from SSIMLoss_weighted import SSIMLoss

class MSESSIMLoss(torch.nn.Module):
    def __init__(self, alpha = 0.7, alpha_weighted=0.5):
        super(MSESSIMLoss, self).__init__()
        self.mse = MSELoss(alpha_weighted = alpha_weighted)
        self.ssim = SSIMLoss(alpha_weighted = alpha_weighted)
        self.alpha = alpha
        self.alpha_weighted = alpha_weighted

    def forward(self, output, mask, ground_truth, error, normalizer):

        # mask = mask.permute(0, 3, 1, 2)

        mse_loss, _, _ = self.mse(output, mask, ground_truth, error, normalizer)
        ssim_loss, _, _ = self.ssim(output, mask, ground_truth, error, normalizer)

        total_loss = self.alpha * mse_loss + (1 - self.alpha) * ssim_loss

        return total_loss, output, ground_truth
