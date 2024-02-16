import torch
import numpy as np
from warnings import warn
from scipy.ndimage import uniform_filter, gaussian_filter

from skimage.util.dtype import dtype_range
from skimage._shared.utils import check_shape_equality

import torch.nn.functional as F

from .processor import BinaryProcessor


class AbsoluteError(BinaryProcessor):
    def __call__(self, image1, image2, allow_in_place=False):
        return abs(image1-image2)

class SquaredError(BinaryProcessor):
    def __call__(self, image1, image2, allow_in_place=False):
        return abs(image1-image2)**2

def structural_similarity(im1, im2,
                          *,
                          win_size=None, data_range=None,
                          gaussian_weights=False,
                          **kwargs):
   
    check_shape_equality(im1, im2)


    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)

    if gaussian_weights:
        # Set to give an 11-tap filter with the default sigma of 1.5 to match
        # Wang et. al. 2004.
        truncate = 3.5

    if win_size is None:
        if gaussian_weights:
            # set win_size used by crop to match the filter size
            r = int(truncate * sigma + 0.5)  # radius as in ndimage
            win_size = 2 * r + 1
        else:
            win_size = 7   # backwards compatibility

    if np.any((np.asarray(im1.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent.  If the input is a multichannel "
            "(color) image, set multichannel=True.")

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        if im1.dtype != im2.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "im1.dtype.", stacklevel=2)
        dmin, dmax = dtype_range[im1.dtype.type]
        data_range = dmax - dmin

    ndim = im1.ndim

    if gaussian_weights:
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma, 'truncate': truncate}
    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)

    NP = win_size ** ndim

    # filter has already normalized by NP
    cov_norm = NP / (NP - 1)  # sample covariance

    # compute (weighted) means
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(im1 * im1, **filter_args)
    uyy = filter_func(im2 * im2, **filter_args)
    uxy = filter_func(im1 * im2, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    return S

# Applies SSIM to last two dimensions
class SSIM2D(BinaryProcessor):
    def __init__(self, win_size=7, k1=0.01, k2=0.03, data_range=None):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)
        self.data_range = data_range
    
    # image2 is reference image
    def call_numpy(self, image1, image2, allow_in_place=False):
        sh = image1.shape
        image1 = image1.reshape(-1,sh[-2],sh[-1])
        image2 = image2.reshape(-1,sh[-2],sh[-1])
        
        if self.data_range is None:
            # TODO: This is max per slice, may also want to use per-3D-image max?
            data_range = image2.max(axis=(1,2))
        else:
            data_range = [self.data_range] * image1.shape[0]
            
        ssim = np.empty_like(image1)
        for i in range(image1.shape[0]):
            ssim[i] = structural_similarity(image1[i], image2[i], win_size=self.win_size, data_range=data_range[i])
        return ssim.reshape(sh)
        
    # image2 is reference image
    def call_torch(self, image1, image2, allow_in_place=False):
        sh = image1.shape
        image1 = image1.reshape(-1,1,sh[-2],sh[-1])
        image2 = image2.reshape(-1,1,sh[-2],sh[-1])
        
        w = torch.ones(1, 1, self.win_size, self.win_size, device=image1.device, dtype=image1.dtype) / self.win_size ** 2

        data_range = self.data_range
        if self.data_range is None:
            # TODO: dim=[2,3]?
            data_range = image2.amax(dim=list(range(1,image2.ndim)), keepdims=True)

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        
        pad = (self.win_size - 1) // 2

        # TODO: Implement this for numpy, shared implementation
        ux = F.conv2d(F.pad(image1, mode='reflect', pad=[pad,pad]*2), w)
        uy = F.conv2d(F.pad(image2, mode='reflect', pad=[pad,pad]*2), w)
        uxx = F.conv2d(F.pad(image1 * image1, mode='reflect', pad=[pad,pad]*2), w)
        uyy = F.conv2d(F.pad(image2 * image2, mode='reflect', pad=[pad,pad]*2), w)
        uxy = F.conv2d(F.pad(image1 * image2, mode='reflect', pad=[pad,pad]*2), w)

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
        
        return S.reshape(sh)
