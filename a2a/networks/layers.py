import torch
import torch.nn as nn
import torch.nn.functional as F

from . import normalization, activation, loss, optimizer

import math


activation('ReLU')(nn.ReLU)
activation('PReLU')(nn.PReLU)
activation('Sigmoid')(nn.Sigmoid)
activation('Tanh')(nn.Tanh)
activation('ELU')(nn.ELU)
activation('SELU')(nn.SELU)
activation('LeakyReLU')(nn.LeakyReLU)
activation('CELU')(nn.CELU)

normalization('BatchNorm1D')(nn.BatchNorm1d)
normalization('BatchNorm2D')(nn.BatchNorm2d)
normalization('BatchNorm3D')(nn.BatchNorm3d)
normalization('InstanceNorm1D')(nn.InstanceNorm1d)
normalization('InstanceNorm2D')(nn.InstanceNorm2d)
normalization('InstanceNorm3D')(nn.InstanceNorm3d)

loss('L1Loss')(nn.L1Loss)
loss('SmoothL1Loss')(nn.SmoothL1Loss)
loss('MSELoss')(nn.MSELoss)
loss('BCELoss')(nn.BCELoss)
loss('BCEWithLogitsLoss')(nn.BCEWithLogitsLoss)
loss('KLDivLoss')(nn.KLDivLoss)
loss('CrossEntropyLoss')(nn.CrossEntropyLoss)

optimizer('SGD')(torch.optim.SGD)
optimizer('Adam')(torch.optim.Adam)
optimizer('AdamW')(torch.optim.AdamW)
optimizer('Rprop')(torch.optim.Rprop)
optimizer('RMSprop')(torch.optim.RMSprop)
optimizer('Adagrad')(torch.optim.Adagrad)
optimizer('Adadelta')(torch.optim.Adadelta)
optimizer('Adamax')(torch.optim.Adamax)
#optimizer('LBFGS')(torch.optim.LBFGS) # TODO: How to implement optimizers with closures?
optimizer('ASGD')(torch.optim.ASGD)



@normalization('PixelNormalization2D')
class PixelNormalization2D(nn.Module):
    def __init__(self, n_features, epsilon=1e-12):
        super().__init__()
        
        self.weight = nn.Parameter(torch.ones((1,n_features,1,1)))
        self.bias = nn.Parameter(torch.zeros((1,n_features,1,1)))
        
        self.epsilon = epsilon
        
    def forward(self, x):
        x = x - x.mean(dim=1, keepdim=True).expand_as(x)
        x = x / (x.norm(p=2,dim=1,keepdim=True) / math.sqrt(x.shape[1] - 1)).clamp(min=self.epsilon)
        
        return x * self.weight.expand_as(x) + self.bias.expand_as(x)

@loss('SSIMLoss')
class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, data_range=None):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)
        self.data_range = data_range

    #def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
    def forward(self, X, Y):
        w = torch.ones(1, 1, self.win_size, self.win_size, device=X.device, dtype=X.dtype) / self.win_size ** 2

        data_range = self.data_range
        if self.data_range is None:
            data_range = Y.amax(dim=list(range(1,Y.ndim)), keepdims=True)

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        
        # ux = F.conv2d(X, self.w)  # typing: ignore
        # uy = F.conv2d(Y, self.w)  #
        # uxx = F.conv2d(X * X, self.w)
        # uyy = F.conv2d(Y * Y, self.w)
        # uxy = F.conv2d(X * Y, self.w)
        
        pad = (self.win_size - 1) // 2
        ux = F.conv2d(F.pad(X, mode='reflect', pad=[pad,pad]*2), w)  # typing: ignore
        uy = F.conv2d(F.pad(Y, mode='reflect', pad=[pad,pad]*2), w)  #
        uxx = F.conv2d(F.pad(X * X, mode='reflect', pad=[pad,pad]*2), w)
        uyy = F.conv2d(F.pad(Y * Y, mode='reflect', pad=[pad,pad]*2), w)
        uxy = F.conv2d(F.pad(X * Y, mode='reflect', pad=[pad,pad]*2), w)
        
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        
        return 1 - S[:,:,pad:-pad,pad:-pad].mean()
