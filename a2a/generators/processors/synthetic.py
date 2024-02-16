from .processor import Processor
from a2a.generators.factory import processor

from a2a.processors.misc import GenerateNoise as GN
from a2a.processors.misc import GenerateNormalNoise as GNN

@processor('GenerateNoise')
class GenerateNoise(Processor):
    def __init__(self, output_dataset=None, batch_size=1, shape=[], dtype='float32', vmin=0, vmax=1, torch=False, device='cuda:0'):
        super().__init__(dataset='')
        
        self.op = GN(shape=[batch_size] + list(shape), vmin=vmin, vmax=vmax, dtype=dtype, torch=torch, device=device)
        self.output_dataset = output_dataset
        self.shape = shape
        self.batch_size = batch_size
    
    def __call__(self, data, attrs):
        modify_shape = False
        if type(self.batch_size) is str:
            batch_size = data[self.batch_size].shape[0]
            modify_shape = True
        else:
            batch_size = self.batch_size
            
        if type(self.shape) is str:
            shape = list(data[self.shape].shape[1:])
            modify_shape = True
        else:
            shape = list(self.shape)
        
        if modify_shape:
            self.op.shape = [batch_size] + shape
        
        data[self.output_dataset] = self.op()

@processor('GenerateNormalNoise')
class GenerateNormalNoise(Processor):
    def __init__(self, output_dataset=None, batch_size=1, shape=[], dtype='float32', sigma=1, mean=0, torch=False, device='cuda:0'):
        super().__init__(dataset='')
        
        self.op = GNN(shape=[batch_size] + list(shape), mean=mean, sigma=sigma, dtype=dtype, torch=torch, device=device)
        self.output_dataset = output_dataset
        self.shape = shape
        self.batch_size = batch_size
    
    def __call__(self, data, attrs):
        modify_shape = False
        if type(self.batch_size) is str:
            batch_size = data[self.batch_size].shape[0]
            modify_shape = True
        else:
            batch_size = self.batch_size
            
        if type(self.shape) is str:
            shape = list(data[self.shape].shape[1:])
            modify_shape = True
        else:
            shape = list(self.shape)
        
        if modify_shape:
            self.op.shape = [batch_size] + shape
        
        data[self.output_dataset] = self.op()

