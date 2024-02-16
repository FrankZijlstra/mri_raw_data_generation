import numpy as np
import torch

from .processor import Processor
from a2a.generators.factory import processor

from a2a.processors import is_numpy

# TODO: Add output dataset
from a2a.processors.transform import Affine2D as Affine2DOp
@processor('Affine2D')
class Affine2D(Processor):
    def __init__(self, dataset=[], rotation_range=(0,0), shift_range_y=(0,0), shift_range_x=(0,0), zoom_range=(1,1), pad_value=0, mode='constant', noise_sigma=1, output_shape=None):
        super().__init__(dataset=dataset)

        self.output_shape = output_shape
        if self.output_shape is not None:
            self.output_shape = tuple(self.output_shape)
        self.rotation_range = rotation_range
        self.shift_range_y = shift_range_y
        self.shift_range_x = shift_range_x
        self.pad_value = pad_value
        self.zoom_range = zoom_range
        self.mode = mode
        self.noise_sigma = noise_sigma
    
    
    def __call__(self, data, attrs):
        
        if self.output_shape is None:
            output_shape = data[self.dataset[0]].shape[-2:]
        else:
            output_shape = self.output_shape

        output = {}
        for x in self.dataset:
            if is_numpy(data[x]):
                output[x] = np.empty(data[x].shape[0:2] + output_shape, dtype=data[x].dtype)
            else:
                output[x] = torch.empty(data[x].shape[0:2] + output_shape, dtype=data[x].dtype, device=data[x].device)
        
        # TODO: Torch processor could take entire batch with batch of transform matrices
        for b in range(data[self.dataset[0]].shape[0]):
            ra = np.random.uniform(self.rotation_range[0], self.rotation_range[1])
            shift_x = np.random.uniform(self.shift_range_x[0], self.shift_range_x[1])
            shift_y = np.random.uniform(self.shift_range_y[0], self.shift_range_y[1])
            zoom = np.random.uniform(self.zoom_range[0], self.zoom_range[1])

            for i,x in enumerate(self.dataset):
                if isinstance(self.pad_value, list):
                    pad_value = self.pad_value[i]
                else:
                    pad_value = self.pad_value
                if isinstance(self.mode, list):
                    mode = self.mode[i]
                else:
                    mode = self.mode
                if isinstance(self.noise_sigma, list):
                    noise_sigma = self.noise_sigma[i]
                else:
                    noise_sigma = self.noise_sigma
                    
                op = Affine2DOp(rotation=ra, shift_y=shift_y, shift_x=shift_x, zoom=zoom, pad_value=pad_value, noise_sigma=noise_sigma, output_shape=self.output_shape, mode=mode)

                output[x][b] = op(data[x][[b]])

        
        for x in self.dataset:
            data[x] = output[x]
