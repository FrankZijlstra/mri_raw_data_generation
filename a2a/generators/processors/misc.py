from .processor import Processor, register_op_processor
from a2a.generators.factory import processor

from a2a.processors.misc import ComplexToChannels, ChannelsToComplex, KeepComplexPhase, Threshold, MakeComplex as MC, Noise, NoiseCov as NC, StackDimensions, SplitChannels, Magnitude
from a2a.processors.fourier import FFT2D, IFFT2D
from a2a.processors.misc import Add, Subtract, Multiply, Divide, Scale
from a2a.processors.cropping import CropChannels

import numpy as np
import torch
from a2a.utils.utils import is_numpy

register_op_processor('ComplexToChannels', ComplexToChannels)
register_op_processor('ChannelsToComplex', ChannelsToComplex)
register_op_processor('Magnitude', Magnitude)
register_op_processor('Add', Add)
register_op_processor('Subtract', Subtract)
register_op_processor('Multiply', Multiply)
register_op_processor('Divide', Divide)
register_op_processor('Scale', Scale)
register_op_processor('CropChannels', CropChannels)

register_op_processor('KeepComplexPhase', KeepComplexPhase)
register_op_processor('Noise', Noise)
register_op_processor('FFT2D', FFT2D)
register_op_processor('IFFT2D', IFFT2D)
register_op_processor('StackDimensions', StackDimensions)
register_op_processor('SplitChannels', SplitChannels)
register_op_processor('Threshold', Threshold)


@processor('PrintShapes')
class PrintShapes(Processor):
    def __init__(self, dataset=[], output_dataset=None):
        super().__init__(dataset=dataset, output_dataset=output_dataset)
        
    def __call__(self, data, attrs):
        for x in data:
            print(x, data[x].shape)


@processor('Concatenate')
class Concatenate(Processor):
    def __init__(self, dataset=[], output_dataset=None):
        super().__init__(dataset=dataset, output_dataset=output_dataset)

    def __call__(self, data, attrs):
        if is_numpy(data[self.dataset[0]]):
            data[self.output_dataset[0]] = np.concatenate(tuple(data[x] for x in self.dataset),axis=1)
        else:
            data[self.output_dataset[0]] = torch.cat(tuple(data[x] for x in self.dataset),axis=1)


@processor('MakeComplex')
class MakeComplex(Processor):
    def __init__(self, dataset=[], output_dataset=None, dataset_phase='', delete_phase=True):
        super().__init__(dataset=dataset, output_dataset=output_dataset)
        self.dataset_phase = dataset_phase
        self.delete_phase = delete_phase
        
        self.op = MC()
        
    def __call__(self, data, attrs):
        im = data[self.dataset[0]]
        phase = data[self.dataset_phase]
        
        data[self.output_dataset[0]] = self.op(im, phase)

        if self.delete_phase:
            del data[self.dataset_phase]
