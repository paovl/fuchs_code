"""
Created on Wed Fen 12 09:29:16 2025

@author: mariagonzalezgarcia
"""

import torch
import torch.nn.functional as F
import numpy as np
import pdb

class NCC_metric(torch.nn.Module):
    def __init__(self, eps = 1e-8, reduction = 'mean'):
        super(NCC_metric, self).__init__()
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
        b, c, h, w = shape

        x = x * mask
        y = y * mask

        idx_0 = torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 0]
        idx_1 = torch.nonzero(mask.sum(dim=(2,3)) > 0)[:, 1]

        number = (mask.sum(dim=(2,3)) > 0).sum(dim=1)[0].item()

        x = x[idx_0, idx_1, :, :]
        y = y[idx_0, idx_1, :, :]

        x = x.view(b, number, h, w)
        y = y.view(b, number, h, w)

        mask = mask[idx_0, idx_1, :, :]

        mask = mask.view(b, number, h, w)

        valid_count = mask.sum(dim=(1, 2,3), keepdim=True)

        # mean
        x_mean = torch.sum(x, dim=(1, 2,3), keepdim=True) / valid_count
        y_mean = torch.sum(y, dim=(1, 2,3), keepdim=True) / valid_count

        # deviation
        x = (x - x_mean) * mask
        y = (y - y_mean) * mask

        dev_xy = torch.mul(x,y)
        dev_xx = torch.mul(x,x)
        dev_yy = torch.mul(y,y)

        dev_xy_sum = torch.sum(dev_xy, dim=(1,2,3))
        dev_xx_sum = torch.sum(dev_xx, dim=(1,2,3))
        dev_yy_sum = torch.sum(dev_yy, dim=(1,2,3))

        ncc = torch.div(dev_xy_sum + eps,
                        torch.sqrt( torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
        
        return ncc

    def forward(self, output, mask, ground_truth, normalizer):

        x = output
        y = ground_truth

        # Calcular la NCC con la función de Pytorch
        ncc= self.normalized_cross_correlation(x, y, mask, reduction='none', eps=1e-8)

        # sum_sample = mask.sum(dim=(1,2,3))

        # map_sum_sample = ncc.sum(dim=(1,2,3))

        # ncc = map_sum_sample / sum_sample

        if self._reduction == 'mean':
            ncc_metric = ncc.mean()
        else:
            ncc_metric = ncc

        # Calcular la pérdida
        return ncc_metric, output, ground_truth

