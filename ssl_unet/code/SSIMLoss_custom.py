"""
Created on Wed Fen 12 09:29:16 2025

@author: mariagonzalezgarcia
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from math import exp
from skimage.metrics import structural_similarity as ssim

class SSIMLoss(torch.nn.Module):
    def __init__(self, kernel_size =(7, 7), sigma = (1.5, 1.5), k1 = 0.01, k2 = 0.03, data_range = 1, channel = 1, device = "cuda:0", size_average = True):
        super(SSIMLoss, self).__init__()
        self.device = device

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range
        self.channel = channel
        self.kernel = self.create_kernel(self.kernel_size, self.sigma)
        self.kernel = self.kernel.expand(self.channel, 1, -1, -1)
        self.size_average = size_average
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self.C1 = (self.k1 * self.data_range)**2
        self.C2 = (self.k2 * self.data_range)**2

    def gaussian(self, kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=self.device)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)
    
    def create_kernel(self, kernel_size, sigma):
        kernel_x = self.gaussian(self.kernel_size[0], sigma[0])
        kernel_y = self.gaussian(self.kernel_size[1], sigma[1])
        return torch.matmul(kernel_x.t(), kernel_y)

    def forward(self, img1, mask, img2, normalizer):

        b, c, h, w = img1.shape

        img1_new = torch.clamp(normalizer.denormalize(img1), 0, 1)[mask]
        img2_new = torch.clamp(normalizer.denormalize(img2), 0, 1)[mask]

        img1_new = img1_new.view(-1, 1, h, w)
        img2_new = img2_new.view(-1, 1, h, w)

        # img1_new = F.pad(img1_new, (self.pad_h, self.pad_h, self.pad_w, self.pad_w), mode='reflect')
        # img2_new = F.pad(img2_new, (self.pad_h, self.pad_h, self.pad_w, self.pad_w), mode='reflect')

        # mu1 = F.conv2d(img1_new, self.kernel, groups = self.channel)
        # mu2 = F.conv2d(img2_new, self.kernel, groups = self.channel)

        # mu1_sq = mu1.pow(2)
        # mu2_sq = mu2.pow(2)
        # mu1_mu2 = mu1*mu2

        # sigma1_sq = F.conv2d(img1_new*img1_new, self.kernel, groups = self.channel) - mu1_sq
        # sigma2_sq = F.conv2d(img2_new*img2_new, self.kernel, groups = self.channel) - mu2_sq
        # sigma12 = F.conv2d(img1_new*img2_new, self.kernel, groups = self.channel) - mu1_mu2

        # a1 = 2*mu1_mu2 + self.C1
        # a2 = 2*sigma12 + self.C2
        # b1 = mu1_sq + mu2_sq + self.C1
        # b2 = sigma1_sq + sigma2_sq + self.C2

        # ssim_idx = (a1 * a2) / (b1 * b2)

        ssim_idx = ssim(np.array(img1_new.detach().cpu()), np.array(img2_new.detach().cpu()), gaussian_weights=True, channel_axis = 1, data_range = 1).item()

        # if self.size_average:
        #     return 1 - ssim_idx.mean(), img1, img2
        # else:
        #     return 1 - ssim_idx.mean(1).mean(1).mean(1), img1, img2
        
        return 1 - ssim_idx, img1, img2

if __name__ == '__main__':
    ssim_loss = SSIMLoss()
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    mask = torch.rand(1, 3, 256, 256)

    img1 = img1.to('cuda:0')
    img2 = img2.to('cuda:0')
    mask = mask.to('cuda:0')

    print(ssim_loss.forward(img1, mask, img2)[0])