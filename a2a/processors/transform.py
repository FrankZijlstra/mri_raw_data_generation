import math
from scipy.ndimage.interpolation import affine_transform, zoom
import torch
import numpy as np

from .processor import UnaryProcessor


# TODO: Allow -1 for dimensions to keep?
class Resample(UnaryProcessor):
    def __init__(self, target_shape=[], order=1):
        super().__init__()
        self.target_shape = target_shape
        self.order = 1
        self.ndim = len(target_shape)
        if self.ndim == 2:
            self.modes = ['nearest', 'bilinear','bicubic']
        elif self.ndim == 3:
            self.modes = ['nearest', 'trilinear']
        else:
            self.modes = ['nearest']
    
    def call_numpy(self, image, allow_in_place=False):
        return zoom(image, [1]*(image.ndim - self.ndim) + list(self.target_shape/np.array(image.shape[-self.ndim:])), order=self.order)

    def call_torch(self, image, allow_in_place=False):
        # TODO: Some reshapes may be required (i.e. 2D resample of 3D matrix should be shaped to 4D)
        if torch.is_complex(image):
            return torch.nn.functional.interpolate(image.real, scale_factor=list(self.target_shape/np.array(image.shape[-self.ndim:])), mode=self.modes[self.order]) + \
                   1j * torch.nn.functional.interpolate(image.imag, scale_factor=list(self.target_shape/np.array(image.shape[-self.ndim:])), mode=self.modes[self.order])

        else:
            return torch.nn.functional.interpolate(image, scale_factor=list(self.target_shape/np.array(image.shape[-self.ndim:])), mode=self.modes[self.order])

class ResampleAvg(UnaryProcessor):
    def __init__(self, target_shape=[]):
        super().__init__()
        self.target_shape = target_shape
        self.ndim = len(target_shape)
    
    # def call_numpy(self, image, allow_in_place=False):
    #     return zoom(image, [1]*(image.ndim - self.ndim) + list(self.target_shape/np.array(image.shape[-self.ndim:])), order=self.order)

    def call_torch(self, image, allow_in_place=False):
        # TODO: Some reshapes may be required (i.e. 2D resample of 3D matrix should be shaped to 4D)
        if image.ndim - self.ndim < 2:
            if torch.is_complex(image):
                return torch.nn.functional.avg_pool2d(image[None,...].real, list(np.int16(np.array(image.shape[-self.ndim:])/self.target_shape)))[0] + \
                       1j * torch.nn.functional.avg_pool2d(image[None,...].imag, list(np.int16(np.array(image.shape[-self.ndim:])/self.target_shape)))[0]
            else:
                return torch.nn.functional.avg_pool2d(image[None,...], list(np.int16(np.array(image.shape[-self.ndim:])/self.target_shape)))[0]
            
        else:
            if torch.is_complex(image):
                return torch.nn.functional.avg_pool2d(image.real, list(np.int16(np.array(image.shape[-self.ndim:])/self.target_shape))) + \
                       1j * torch.nn.functional.avg_pool2d(image.imag, list(np.int16(np.array(image.shape[-self.ndim:])/self.target_shape)))
            else:
                return torch.nn.functional.avg_pool2d(image, list(np.int16(np.array(image.shape[-self.ndim:])/self.target_shape)))
            
# TODO: Affine3D
# TODO: Figure out how to handle batch torch operation with multiple parameters
class Affine2D(UnaryProcessor):
    def __init__(self, rotation=0, shift_y=0, shift_x=0, zoom=1, pad_value=0, noise_sigma=1, mode='constant', output_shape=None):
        super().__init__()

        self.rotation = rotation
        self.shift_y = shift_y
        self.shift_x = shift_x
        self.zoom = zoom
        self.output_shape = output_shape
        self.pad_value = pad_value
        self.noise_sigma = noise_sigma
        self.mode = mode
        self.mode_torch = {'reflect':'reflection', 'constant':'zeros', 'noise':'zeros', 'border':'border'}[mode]
        
        if self.mode_torch == 'zeros' and pad_value != 0:
            raise ValueError('Torch only supports constant padding with 0')

    def transform_numpy(self, input_shape, output_shape):
        transform = np.array([[math.cos(self.rotation/180*math.pi)*self.zoom, -math.sin(self.rotation/180*math.pi)*self.zoom],
                              [math.sin(self.rotation/180*math.pi)*self.zoom,  math.cos(self.rotation/180*math.pi)*self.zoom]])
        
        c_in = 0.5*np.array(input_shape)
        c_out = 0.5*np.array(output_shape) + np.array([self.shift_y, self.shift_x])
        offset = c_in - c_out.dot(transform)
        
        return transform, offset
    
    def transform_torch(self, input_shape, output_shape):
        # TODO: Handle output_shape? Shift?
        transform = torch.Tensor([[math.cos(self.rotation/180*math.pi)*self.zoom, -math.sin(self.rotation/180*math.pi)*self.zoom, self.shift_y*self.zoom/input_shape[0]],
                                  [math.sin(self.rotation/180*math.pi)*self.zoom,  math.cos(self.rotation/180*math.pi)*self.zoom, self.shift_x*self.zoom/input_shape[1]]])
                
        return transform
    
    def call_numpy(self, image, allow_in_place=False):
        input_shape = image.shape[-2:]
        if self.output_shape is None:
            output_shape = image.shape[-2:]
        else:
            output_shape = self.output_shape
        
        transform, offset = self.transform_numpy(input_shape, output_shape)

        o_shape = image.shape[:-2] + output_shape
        image = image.reshape((-1,) + image.shape[-2:])
        output = np.empty((image.shape[0],) + output_shape, dtype=image.dtype)

        for i in range(image.shape[0]):
            affine_transform(image[i], transform.T, offset=offset, output=output[i], output_shape=output_shape, order=1, cval=self.pad_value, mode=self.mode)
        
        return output.reshape(o_shape)
    
    def call_torch(self, image, allow_in_place=False):
        input_shape = image.shape[-2:]
        if self.output_shape is None:
            output_shape = image.shape[-2:]
        else:
            output_shape = self.output_shape
            
        transform = self.transform_torch(input_shape, (1,1) + output_shape)

        if torch.is_complex(image):
            transform = transform.to(image.device, dtype=image.real.dtype, non_blocking=True)
            grid = torch.nn.functional.affine_grid(transform[None,:,:], (1,1) + output_shape)
            output = torch.nn.functional.grid_sample(image.real, grid, padding_mode=self.mode_torch) + 1j*torch.nn.functional.grid_sample(image.imag, grid, padding_mode=self.mode_torch)
            
            if self.mode == 'noise':
                noise = torch.randn_like(output) * self.noise_sigma
                output[output == 0] = noise[output == 0]
            
        else:
            transform = transform.to(image.device, dtype=image.dtype, non_blocking=True)

            grid = torch.nn.functional.affine_grid(transform[None,:,:], (1,1) + output_shape)
            output = torch.nn.functional.grid_sample(image, grid, padding_mode=self.mode_torch)
            
            if self.mode == 'noise':
                noise = torch.randn_like(output) * self.noise_sigma
                output[output == 0] = noise[output == 0]

        return output[0]
