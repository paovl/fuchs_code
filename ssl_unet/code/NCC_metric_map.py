"""
Created on Wed Fen 12 09:29:16 2025

@author: mariagonzalezgarcia
"""

import torch
import torch.nn.functional as F
import numpy as np
import pdb

class NCC_metric_map(torch.nn.Module):
    def __init__(self, eps = 1e-8, reduction = 'mean'):
        super(NCC_metric_map, self).__init__()
        self._eps = eps
        self._reduction = reduction


    def normalized_cross_correlation(self, x, y, mask, reduction='mean', eps=1e-8):
        """ N-dimensional normalized cross correlation (NCC)

        Args:
            x (~torch.Tensor): Input tensor.
            y (~torch.Tensor): Input tensor.
            return_map (bool): If True, also return the correlation map.
            reduction (str, optional): Specifies the reduction to apply to the output:
                ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.

        Returns:
            ~torch.Tensor: Output scalar
            ~torch.Tensor: Output tensor
        """
        shape = x.shape
        c, h, w = shape

        x = x * mask
        y = y * mask

        valid_count = mask.sum()

        # mean
        x_mean = x.sum() / valid_count
        y_mean = y.sum() / valid_count

        # deviation
        x = (x - x_mean) * mask
        y = (y - y_mean) * mask

        dev_xy = x * y
        dev_xx = x * x
        dev_yy = y * y

        dev_xx_sum = dev_xx.sum()
        dev_yy_sum = dev_yy.sum()

        # lo mismo pero para numpy
        ncc  = (dev_xy + eps) / (np.sqrt(dev_xx * dev_yy) + eps)

        return ncc

    def forward(self, output, mask, ground_truth):

        x = output
        y = ground_truth

        # Calcular la NCC con la función de Pytorch
        ncc= self.normalized_cross_correlation(x, y, mask, reduction='none', eps=1e-8)

        # sum_sample = mask.sum(dim=(1,2,3))

        # map_sum_sample = ncc.sum(dim=(1,2,3))

        # ncc = map_sum_sample / sum_sample

        # Calcular la pérdida
        return ncc, output, ground_truth

