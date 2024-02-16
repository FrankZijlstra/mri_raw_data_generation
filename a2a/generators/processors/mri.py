from .processor import Processor, register_op_processor
from a2a.generators.factory import processor

from a2a.processors.misc import NoiseCov as NC

import numpy as np
import torch
from a2a.utils.utils import is_numpy

@processor('NoiseCov')
class NoiseCov(Processor):
    def __init__(self, dataset=[], output_dataset=None, sigma=1, scale=1):
        super().__init__(dataset=dataset, output_dataset=output_dataset)
        self.sigma = sigma
        self.scale = scale
        self.op = NC(sigma)
        
    def __call__(self, data, attrs):
        output = {}
        for x in self.dataset:
            if is_numpy(data[x]):
                output[x] = np.empty_like(data[x])
            else:
                output[x] = torch.empty_like(data[x])
        
        for b in range(data[self.dataset[0]].shape[0]):

            if isinstance(self.sigma, str):
                if data[self.sigma].shape[0] == 1:
                    self.op.sigma = data[self.sigma][0,0] * self.scale
                else:
                    self.op.sigma = data[self.sigma][b,0] * self.scale
            else:
                self.op.sigma = self.sigma
            
            for i,x in enumerate(self.dataset):
                output[x][b] = self.op(data[x][[b]])

        for o,x in zip(self.output_dataset, self.dataset):
            data[o] = output[x]