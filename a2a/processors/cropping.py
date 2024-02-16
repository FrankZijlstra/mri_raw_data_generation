import torch
import numpy as np

from .processor import UnaryProcessor, is_numpy

class CropChannels(UnaryProcessor):
    def __init__(self, start=None, stop=None, channel_dim=1):
        super().__init__()
        self.start = start
        self.stop = stop
        self.channel_dim = channel_dim
    
    def __call__(self, image, allow_in_place=False):
        indexes = (slice(None),) * (self.channel_dim) + (slice(self.start, self.stop),)
        return image[indexes]

class Crop2D(UnaryProcessor):
    def __init__(self, size_y=None, size_x=None):
        super().__init__()
        self.size_y = size_y
        self.size_x = size_x
    
    def __call__(self, image, allow_in_place=False):
        s = image.shape
        indexes = (slice(None),) * (image.ndim - 2) + (
                   slice(s[-2]//2 - self.size_y//2,s[-2]//2 - self.size_y//2 + self.size_y) if self.size_y is not None else slice(None),
                   slice(s[-1]//2 - self.size_x//2,s[-1]//2 - self.size_x//2 + self.size_x) if self.size_x is not None else slice(None))
        return image[indexes]

class Crop3D(UnaryProcessor):
    def __init__(self, size_z=None, size_y=None, size_x=None):
        super().__init__()
        self.size_z = size_z
        self.size_y = size_y
        self.size_x = size_x
    
    def __call__(self, image, allow_in_place=False):
        s = image.shape
        indexes = (slice(None),) * (image.ndim - 3) + (
                   slice(s[-3]//2 - self.size_z//2,s[-3]//2 - self.size_z//2 + self.size_z) if self.size_z is not None else slice(None),
                   slice(s[-2]//2 - self.size_y//2,s[-2]//2 - self.size_y//2 + self.size_y) if self.size_y is not None else slice(None),
                   slice(s[-1]//2 - self.size_x//2,s[-1]//2 - self.size_x//2 + self.size_x) if self.size_x is not None else slice(None))
        return image[indexes]

class Pad2D(UnaryProcessor):
    def __init__(self, size_y=None, size_x=None, pad_value=0):
        super().__init__()
        self.size_y = size_y
        self.size_x = size_x
        self.pad_value = pad_value
        
    def __call__(self, image, allow_in_place=False):
        s = image.shape
        indexes = (slice(None),) * (image.ndim - 2) + (
                   slice(self.size_y//2 - s[-2]//2, self.size_y//2 - s[-2]//2 + s[-2]) if self.size_y is not None else slice(None),
                   slice(self.size_x//2 - s[-1]//2, self.size_x//2 - s[-1]//2 + s[-1]) if self.size_x is not None else slice(None))
        output_shape = image.shape[0:image.ndim-2] + (
                        self.size_y if self.size_y is not None else image.shape[-2],
                        self.size_x if self.size_x is not None else image.shape[-1])
        
        if is_numpy(image):
            output = np.ones(output_shape, dtype=image.dtype) * self.pad_value
        else:
            output = torch.ones(output_shape, dtype=image.dtype, device=image.device) * self.pad_value
        output[indexes] = image
        return output

class Pad3D(UnaryProcessor):
    def __init__(self, size_z=None, size_y=None, size_x=None, pad_value=0):
        super().__init__()
        self.size_z = size_z
        self.size_y = size_y
        self.size_x = size_x
        self.pad_value = pad_value
        
    def __call__(self, image, allow_in_place=False):
        s = image.shape
        indexes = (slice(None),) * (image.ndim - 3) + (
                   slice(self.size_z//2 - s[-3]//2, self.size_z//2 - s[-3]//2 + s[-3]) if self.size_z is not None else slice(None),
                   slice(self.size_y//2 - s[-2]//2, self.size_y//2 - s[-2]//2 + s[-2]) if self.size_y is not None else slice(None),
                   slice(self.size_x//2 - s[-1]//2, self.size_x//2 - s[-1]//2 + s[-1]) if self.size_x is not None else slice(None))
        output_shape = image.shape[0:image.ndim-3] + (
                        self.size_z if self.size_z is not None else image.shape[-3],
                        self.size_y if self.size_y is not None else image.shape[-2],
                        self.size_x if self.size_x is not None else image.shape[-1])
        
        if is_numpy(image):
            output = np.ones(output_shape, dtype=image.dtype) * self.pad_value
        else:
            output = torch.ones(output_shape, dtype=image.dtype, device=image.device) * self.pad_value
        output[indexes] = image
        return output
