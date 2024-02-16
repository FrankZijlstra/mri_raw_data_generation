import torch
import numpy as np

from .processor import UnaryProcessor

class FFT2D(UnaryProcessor):
    def __init__(self):
        super().__init__()
        
    def call_numpy(self, image, allow_in_place=False):
        x = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=(-2,-1)), axes=(-2,-1), norm='ortho'), axes=(-2,-1))
        if image.dtype == np.complex64 or image.dtype == np.float32:
            x = x.astype(np.complex64)
        return x

    def call_torch(self, image, allow_in_place=False):
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2,-1)), dim=(-2,-1), norm='ortho'), dim=(-2,-1))

class IFFT2D(UnaryProcessor):
    def __init__(self):
        super().__init__()
        
    def call_numpy(self, image, allow_in_place=False):
        x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(image, axes=(-2,-1)), axes=(-2,-1), norm='ortho'), axes=(-2,-1))
        if image.dtype == np.complex64 or image.dtype == np.float32:
            x = x.astype(np.complex64)
        return x
    
    def call_torch(self, image, allow_in_place=False):
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(image, dim=(-2,-1)), dim=(-2,-1), norm='ortho'), dim=(-2,-1))
