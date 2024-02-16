from a2a.generators.factory import processor
from .processor import Processor

import random
import numpy as np
import torch

from a2a.processors.transform import Resample as ResampleOp
from a2a.processors.transform import ResampleAvg as ResampleAvgOp

from a2a.processors.misc import TransposeFlip, PixelShift2D
from a2a.processors.transform import Affine2D

from a2a.processors import is_numpy


@processor('DataAugmentation2D')
class DataAugmentation2D(Processor):

    def __init__(self, dataset=[], output_dataset=None, probabilities={}, parameters={}, mode='constant', pad_value=0, noise_sigma=0, output_shape=None):
        super().__init__(dataset=dataset, output_dataset=output_dataset)
        
        self.probabilities = {'rotation': 0.5,
                              'rotation_90': 0.5,
                              'translation': 0.5,
                              'zoom': 0.5,
                              'pixel_translation': 0.5,
                              'flip': 0.5}
        self.probabilities.update(probabilities)
        
        self.parameters = {'rotation': (-10, 10),
                           'rotation_90': [0, 90, 180, 270],
                           'translation': ((-16, 16), (-16, 16)),
                           'zoom': (0.9, 1.1),
                           'pixel_translation': ((-8, 8), (-8,8)),
                           'flip': 'xy'
                           }
        self.parameters.update(parameters)
        
        self.affine_first = False
        self.pad_value = pad_value
        self.mode = mode
        self.output_shape = output_shape
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
        
        for b in range(data[self.dataset[0]].shape[0]):
            use_affine = False
            # TODO: Check parameter ranges (i.e. zoom (1,1) -> use_affine = False)
            if random.random() < self.probabilities['rotation']:
                ra = random.uniform(self.parameters['rotation'][0], self.parameters['rotation'][1])
                use_affine = True
            else:
                ra = 0
            
            if random.random() < self.probabilities['translation']:
                shift_x = random.uniform(self.parameters['translation'][1][0], self.parameters['translation'][1][1])
                shift_y = random.uniform(self.parameters['translation'][0][0], self.parameters['translation'][0][1])
                use_affine = True
            else:
                shift_x = 0
                shift_y = 0
            
            if random.random() < self.probabilities['zoom']:
                zoom = random.uniform(self.parameters['zoom'][0], self.parameters['zoom'][1])
                use_affine = True
            else:
                zoom = 1
            
            if random.random() < self.probabilities['rotation_90']:
                ra_90 = random.choice(self.parameters['rotation_90'])
            else:
                ra_90 = 0
            
            if random.random() < self.probabilities['flip']:
                flip_x = random.random() < 0.5 if 'x' in self.parameters['flip'] else False
                flip_y = random.random() < 0.5 if 'y' in self.parameters['flip'] else False
            else:
                flip_x = False
                flip_y = False
            
            transpose = False
            if ra_90 == 90:
                flip_x = not flip_x
                transpose = True
            elif ra_90 == 180:
                flip_x = not flip_x
                flip_y = not flip_y
            elif ra_90 == 270:
                flip_y = not flip_y
                transpose = True
            
            flip = []
            if flip_y:
                flip.append(0)
            if flip_x:
                flip.append(1)
            
            if random.random() < self.probabilities['pixel_translation']:
                pixelshift_x = random.randint(self.parameters['pixel_translation'][1][0], self.parameters['pixel_translation'][1][1])
                pixelshift_y = random.randint(self.parameters['pixel_translation'][0][0], self.parameters['pixel_translation'][0][1])
            else:
                pixelshift_x = 0
                pixelshift_y = 0
            
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
                elif isinstance(self.noise_sigma, str):
                    if data[self.noise_sigma].shape[0] == 1:
                        noise_sigma = data[self.noise_sigma][0,0]
                    else:
                        noise_sigma = data[self.noise_sigma][b,0]
                else:
                    noise_sigma = self.noise_sigma
                    
                if pixelshift_y != 0 or pixelshift_x != 0:
                    op1 = PixelShift2D(shift_y=pixelshift_y, shift_x=pixelshift_x)
                else:
                    op1 = lambda x: x
                
                if transpose or len(flip) > 0:
                    op2 = TransposeFlip(transpose=[1,0] if transpose else [0,1], flip=flip)
                else:
                    op2 = lambda x: x
                
                if use_affine:
                    op3 = Affine2D(rotation=ra, shift_y=shift_y, shift_x=shift_x, zoom=zoom, pad_value=pad_value, noise_sigma=noise_sigma, output_shape=self.output_shape, mode=mode)
                else:
                    op3 = lambda x: x
                    
                output[x][b] = op3(op2(op1(data[x][[b]])))
        for o,x in zip(self.output_dataset, self.dataset):
            data[o] = output[x]


@processor('Resample')
class Resample(Processor):
    def __init__(self, dataset=[], output_dataset=None, target_shape=[], order=1):
        super().__init__(dataset=dataset, output_dataset=output_dataset)
        self.target_shape = target_shape
        if type(target_shape) is str or type(target_shape) is float:
            target_shape = [1,1]
        self.op = ResampleOp(target_shape=target_shape, order=order)
    
    def __call__(self, data, attrs):
        if type(self.target_shape) is float:
            self.op.target_shape = tuple(round(y*self.target_shape) for y in data[self.dataset[0]].shape[2:])
        if type(self.target_shape) is str:
            self.op.target_shape = data[self.target_shape].shape[2:]

        for o,x in zip(self.output_dataset, self.dataset):
            data[o] = self.op(data[x])

@processor('ResampleAvg')
class ResampleAvg(Processor):
    def __init__(self, dataset=[], output_dataset=None, target_shape=[]):
        super().__init__(dataset=dataset, output_dataset=output_dataset)
        self.target_shape = target_shape
        if type(target_shape) is str:
            target_shape = [1,1]
        self.op = ResampleAvgOp(target_shape=target_shape)
    
    def __call__(self, data, attrs):
        if type(self.target_shape) is str:
            self.op.target_shape = data[self.target_shape].shape[2:]

        for o,x in zip(self.output_dataset, self.dataset):
            data[o] = self.op(data[x])
