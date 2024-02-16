import numpy as np
import scipy.ndimage

from .preprocessor_base import Preprocessor
from . import preprocessor

@preprocessor('CreateMask')
class CreateMask(Preprocessor):
    def __init__(self, name=None, dataset=[], mask_name='mask', channel_index=0, threshold=0.2, threshold_absolute=False, fill_holes=True, stored_parameters=None):
        super().__init__(name=name, dataset=dataset, stored_parameters=stored_parameters)
        
        self.mask_name = mask_name
        self.channel_index = channel_index
        self.threshold = threshold
        self.threshold_absolute = threshold_absolute
        self.fill_holes = fill_holes
    
    def process(self, dataset, attrs):
        if self.mask_name in dataset:
            dataset[self.mask_name].clear()
            attrs[self.mask_name] = attrs[self.dataset[0]] # TODO: Should this copy?
        else:
            dataset[self.mask_name] = []
            attrs[self.mask_name] = attrs[self.dataset[0]] # TODO: Should this copy?
        
        for x in dataset[self.dataset[0]]:
            if self.threshold_absolute:
                dataset[self.mask_name].append((x[self.channel_index,:,:,:] > self.threshold)[np.newaxis,...])
            else:
                m = x[self.channel_index].mean()
                m = x[self.channel_index][x[self.channel_index] > m].mean() * 0.2
                dataset[self.mask_name].append((x[self.channel_index,:,:,:] > m)[np.newaxis,...])
            
            if self.fill_holes:
                mask = dataset[self.mask_name][-1][0]
                for j in range(mask.shape[0]):
                    mask[j,:,:] = scipy.ndimage.morphology.binary_fill_holes(mask[j,:,:])
                for j in range(mask.shape[1]):
                    mask[:,j,:] = scipy.ndimage.morphology.binary_fill_holes(mask[:,j,:])
                for j in range(mask.shape[2]):
                    mask[:,:,j] = scipy.ndimage.morphology.binary_fill_holes(mask[:,:,j])

        return None
    
    def process_inverse(self, dataset, attrs, inverse_parameters):
        pass
        

# MaskInvalid creates a new mask dataset if it does not exist
@preprocessor('MaskInvalid')
class MaskInvalid(Preprocessor):
    def __init__(self, name=None, dataset=[], mask_name='mask', invalid_value=-2000, stored_parameters=None):
        super().__init__(name=name, dataset=dataset, stored_parameters=stored_parameters)
        
        self.mask_name = mask_name
        self.invalid_value = invalid_value
        
    def process(self, dataset, attrs):
        if self.mask_name not in dataset:
            dataset[self.mask_name] = [np.ones((1,) + x.shape[1:], dtype=np.bool) for x in dataset[self.dataset[0]]]
            attrs[self.mask_name] = attrs[self.dataset[0]]

        for x in self.dataset:
            for i,d in enumerate(dataset[x]):
                dataset[self.mask_name][i][0] &= ~np.any(d == self.invalid_value, axis=0)
                dataset[self.mask_name][i][0] &= ~np.any(np.isnan(d), axis=0)
        
        return None
    
    def process_inverse(self, dataset, attrs, inverse_parameters):
        pass

# MaskThreshold creates a new mask dataset if it does not exist
@preprocessor('MaskThreshold')
class MaskThreshold(Preprocessor):
    def __init__(self, name=None, dataset=[], mask_name='mask', threshold=0, threshold_above=True, stored_parameters=None):
        super().__init__(name=name, dataset=dataset, stored_parameters=stored_parameters)
        
        self.mask_name = mask_name
        self.threshold = threshold
        self.threshold_above = threshold_above
        
    def process(self, dataset, attrs):
        if self.mask_name not in dataset:
            dataset[self.mask_name] = [np.ones((1,) + x.shape[1:], dtype=np.bool) for x in dataset[self.dataset[0]]]
            attrs[self.mask_name] = attrs[self.dataset[0]]

        for x in self.dataset:
            for i,d in enumerate(dataset[x]):
                if self.threshold_above:
                    dataset[self.mask_name][i][0] &= np.all(d >= self.threshold, axis=0)
                else:
                    dataset[self.mask_name][i][0] &= np.all(d < self.threshold, axis=0)
        
        return None
    
    def process_inverse(self, dataset, attrs, inverse_parameters):
        pass

@preprocessor('ApplyMask')
class ApplyMask(Preprocessor):
    def __init__(self, name=None, dataset=[], mask_name='mask', fill_value=0, stored_parameters=None):        
        super().__init__(name=name, dataset=dataset, stored_parameters=stored_parameters)
        
        self.mask_name = mask_name
        self.fill_value = fill_value

    def check_dataset(self, dataset):
        return self.mask_name in dataset and super().check_dataset(dataset)

    def process(self, dataset, attrs):
        for x in self.dataset:
            for i,d in enumerate(dataset[x]):
                dataset[x][i][:,~dataset[self.mask_name][i][0]] = self.fill_value
        return None
    
    def process_inverse(self, dataset, attrs, inverse_parameters):
        pass

