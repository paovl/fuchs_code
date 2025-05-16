"""
Created on Wed Fen 12 09:29:16 2025

@author: mariagonzalezgarcia
"""

import torch
import torch.nn.functional as F
import numpy as np
from MSELoss import MSELoss
from SSIMLoss import SSIMLoss

class MSESSIMLoss(torch.nn.Module):
    def __init__(self, alpha = 0.7):
        super(MSESSIMLoss, self).__init__()
        self.mse = MSELoss()
        self.ssim = SSIMLoss()
        self.alpha = alpha

    def forward(self, output, mask, ground_truth, normalizer):

        # mask = mask.permute(0, 3, 1, 2)

        mse_loss, _, _ = self.mse(output, mask, ground_truth, normalizer)
        ssim_loss, _, _ = self.ssim(output, mask, ground_truth, normalizer)

        total_loss = self.alpha * mse_loss + (1 - self.alpha) * ssim_loss

        return total_loss, output, ground_truth
