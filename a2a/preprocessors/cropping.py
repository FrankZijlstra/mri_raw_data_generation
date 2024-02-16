import numpy as np

from . import preprocessor
from .preprocessor_base import Preprocessor


@preprocessor('Crop')
class Crop(Preprocessor):
    def __init__(self, name=None, dataset=[], size_z=None, size_y=None, size_x=None, pad_value=0, stored_parameters=None):
        super().__init__(name=name, dataset=dataset, stored_parameters=stored_parameters)
        self.size_z = size_z
        self.size_y = size_y
        self.size_x = size_x
        self.pad_value = pad_value
    
    def process(self, dataset, attrs):        
        s = dataset[self.dataset[0]].shape
        # TODO: nD implementation
        indexes = (slice(None),
                   slice(s[1]//2 - self.size_z//2,s[1]//2 - self.size_z//2 + self.size_z) if self.size_z is not None else slice(None),
                   slice(s[2]//2 - self.size_y//2,s[2]//2 - self.size_y//2 + self.size_y) if self.size_y is not None else slice(None),
                   slice(s[3]//2 - self.size_x//2,s[3]//2 - self.size_x//2 + self.size_x) if self.size_x is not None else slice(None))
        
        # TODO: Parameters per dataset?
        inverse_parameters = {'shape':s, 'indexes':indexes}
        
        for x in self.dataset:
            dataset[x] = dataset[x][indexes]
            
        return inverse_parameters

    def process_inverse(self, dataset, attrs, inverse_parameters):
        for di,x in enumerate(self.dataset):
            if isinstance(self.pad_value, list):
                pad_value = self.pad_value[di]
            else:
                pad_value = self.pad_value
            tmp = np.ones(inverse_parameters['shape'], dtype=dataset[x].dtype) * pad_value
            tmp[inverse_parameters['indexes']] = dataset[x]
            dataset[x] = tmp

@preprocessor('Pad')
class Pad(Preprocessor):
    def __init__(self, name=None, dataset=[], shape=None, pad_value=0, stored_parameters=None):
        super().__init__(name=name, dataset=dataset, stored_parameters=stored_parameters)
        self.shape = shape
        self.pad_value = pad_value
    
    def process(self, dataset, attrs):
        target_shape = dataset[self.dataset[0]].shape[:-len(self.shape)] + tuple(self.shape)
        indexes = tuple([slice(tx//2 - x//2, tx//2 - x//2 + x) for tx,x in zip(target_shape, dataset[self.dataset[0]].shape)])
        
        inverse_parameters = {'indexes':indexes}
        
        for di,x in enumerate(self.dataset):
            if isinstance(self.pad_value, list):
                pad_value = self.pad_value[di]
            else:
                pad_value = self.pad_value
            tmp = np.ones(target_shape, dtype=dataset[x].dtype) * pad_value
            tmp[indexes] = dataset[x]
            dataset[x] = tmp
    
        return inverse_parameters

    def process_inverse(self, dataset, attrs, inverse_parameters):
        for di,x in enumerate(self.dataset):
            for i,d in enumerate(dataset[x]):
                dataset[x][i] = dataset[x][i][inverse_parameters[i]['indexes']]


from a2a.processors.processor import is_complex
# Crop is always based on first element of dataset
@preprocessor('CropSlices')
class CropSlices(Preprocessor):
    def __init__(self, name=None, dataset=[], slice_dim=1, threshold=0.2, stored_parameters=None):
        super().__init__(name=name, dataset=dataset, stored_parameters=stored_parameters)
        self.slice_dim = slice_dim
        self.threshold = threshold
    
    def process(self, dataset, attrs):
        d = dataset[self.dataset[0]]
        if is_complex(d):
            d = abs(d)
        
        s = d.shape[1:]
        m = (d-d.min()).mean(axis=tuple([x for x in range(4) if x != self.slice_dim]))
        slice_mask = m>self.threshold*m.mean()

        indexes = [slice(None) for x in range(4)]
        indexes[self.slice_dim] = slice_mask
        indexes = tuple(indexes)
        inverse_parameters = {'dtype':d.dtype, 'shape':s, 'indexes':indexes}
        
        for x in self.dataset:
            dataset[x] = dataset[x][indexes]
            
        return inverse_parameters

    def process_inverse(self, dataset, attrs, inverse_parameters):
        if isinstance(dataset, tuple):
            for x in dataset:
                for i,d in enumerate(x):
                    tmp = np.zeros((d.shape[0],) + inverse_parameters[i]['shape'], dtype=inverse_parameters[i]['dtype'])
                    tmp[inverse_parameters[i]['indexes']] = d
                    x[i] = tmp
        else:
            for i,d in enumerate(dataset):
                tmp = np.zeros((d.shape[0],) + inverse_parameters[i]['shape'], dtype=inverse_parameters[i]['dtype'])
                tmp[inverse_parameters[i]['indexes']] = d
                dataset[i] = tmp
