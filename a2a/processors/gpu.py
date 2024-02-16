import torch
from .processor import UnaryProcessor

class GPU(UnaryProcessor):
    def __init__(self, device='cuda:0', dtype=None):
        super().__init__()
        self.device = device
        
        if type(dtype) is str:
            self.dtype = {'float32':torch.float32,
                          'float64':torch.float64,
                          'complex64':torch.complex64,
                          'complex128':torch.complex128}[dtype]
        else:
            self.dtype = dtype
    
    def call_numpy(self, image, allow_in_place=False):
        t = torch.from_numpy(image)
        if self.dtype:
            return t.to(self.device, dtype=self.dtype, non_blocking=True)
        else:
            return t.to(self.device, non_blocking=True)

    def call_torch(self, image, allow_in_place=False):
        if self.dtype:
            return image.to(self.device, dtype=self.dtype, non_blocking=True)
        else:
            return image.to(self.device, non_blocking=True)

# TODO: Should CPU always be numpy?
class CPU(UnaryProcessor):
    def __init__(self, dtype=None):
        super().__init__()
        self.dtype = dtype
    
    def call_numpy(self, image, allow_in_place=False):
        # t = torch.from_numpy(image)
        # if self.dtype:
        #     return t.to(dtype=self.dtype)
        # else:
        #     return t
        if self.dtype:
            return image.astype(dtype=self.dtype)
        else:
            return image
    
    def call_torch(self, image, allow_in_place=False):
        if self.dtype:
            return image.detach().cpu().numpy().astype(dtype=self.dtype)
        else:
            return image.detach().cpu().numpy()

class Numpy(UnaryProcessor):
    def __init__(self, dtype=None):
        super().__init__()
        self.dtype = dtype
    
    def call_numpy(self, image, allow_in_place=False):
        if self.dtype:
            return image.astype(dtype=self.dtype)
        else:
            return image
    
    def call_torch(self, image, allow_in_place=False):
        if self.dtype:
            return image.detach().cpu().numpy().astype(dtype=self.dtype)
        else:
            return image.detach().cpu().numpy()
