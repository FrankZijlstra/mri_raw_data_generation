from . import model_runner

from .model_runner_base import ModelRunner

import torch
import numpy as np

from a2a.processors import is_numpy

@model_runner('Slice', processor_parameters=['processors'])
class SliceModelRunner(ModelRunner):
    def __init__(self, model, input_dataset=[], batch_size=1, processors=[]):
        super().__init__(model)
        
        if isinstance(input_dataset, str):
            self.input_dataset = [input_dataset]
        else:
            self.input_dataset = input_dataset
        self.batch_size = batch_size
        self.processors = processors


    # subimage should be a valid slice tuple, it will be applied to the result image (shape CxZxYxX)
    # TODO: subimage could be mask
    
    # dataset is dict as received from a dataset (i.e. C + 3D images)
    def __call__(self, dataset, attrs, subimage=None):
        data = {}
        attr = {}

        output = {}
        
        # TODO: Is this right?
        image_shape = dataset[self.input_dataset[0]].shape
        batch_indexes = np.zeros(self.batch_size, dtype=np.int16)
        
        if subimage is None or len(subimage) == 1:
            required_slices = range(image_shape[1])
        else:
            required_slices = range(image_shape[1])[subimage[1]]
            if isinstance(required_slices, int):
                required_slices = [required_slices]
        
        b = 0
        for coord_z in range(image_shape[1]):
            if coord_z not in required_slices:
                continue
            
            im = {x:dataset[x] for x in self.input_dataset}
            for x in self.input_dataset:
                if x not in data:
                    if is_numpy(im[x]):
                        data[x] = np.empty((self.batch_size, im[x].shape[0]) + (im[x].shape[2:]), dtype=im[x].dtype)
                    else:
                        data[x] = torch.empty((self.batch_size, im[x].shape[0]) + (im[x].shape[2:]), dtype=im[x].dtype, device=im[x].device)
                data[x][b] = im[x][:,coord_z]
        
            batch_indexes[b] = coord_z
            
            b += 1
            if b == self.batch_size:
                for processor in self.processors:
                    processor(data, attr)

                out = self.model(data)

                for i in range(b):
                    for x in out:
                        if x not in output:
                            if is_numpy(out[x]):
                                output[x] = np.empty((out[x].shape[1], image_shape[1]) + out[x].shape[2:], dtype=out[x].dtype)
                            else:
                                output[x] = torch.empty((out[x].shape[1], image_shape[1]) + out[x].shape[2:], dtype=out[x].dtype, device=out[x].device)
                        output[x][:,batch_indexes[i]] = out[x][i,:]

                b = 0
                data = {}
                attr = {}
        
        if b > 0:
            data = {x:data[x][0:b] for x in data}
            for processor in self.processors:
                processor(data, attr)
            out = self.model(data)
                
            for i in range(b):
                for x in out:
                    if x not in output:
                        if is_numpy(out[x]):
                            output[x] = np.empty((out[x].shape[1], image_shape[1]) + out[x].shape[2:], dtype=out[x].dtype)
                        else:
                            output[x] = torch.empty((out[x].shape[1], image_shape[1]) + out[x].shape[2:], dtype=out[x].dtype, device=out[x].device)
                    output[x][:,batch_indexes[i]] = out[x][i,:]
            
        if subimage is not None:
            for x in output:
                output[x] = output[x][subimage]
        
        return output

from a2a.utils.utils import get_random_state, set_random_state, set_seed

@model_runner('SliceSeeded', processor_parameters=['processors'])
class SliceSeededModelRunner(ModelRunner):
    def __init__(self, model, input_dataset=[], batch_size=1, processors=[]):
        super().__init__(model)
        
        if isinstance(input_dataset, str):
            self.input_dataset = [input_dataset]
        else:
            self.input_dataset = input_dataset
        self.batch_size = batch_size
        self.processors = processors


    # subimage should be a valid slice tuple, it will be applied to the result image (shape CxZxYxX)
    # TODO: subimage could be mask
    
    # dataset is dict as received from a dataset (i.e. C + 3D images)
    def __call__(self, dataset, attrs, subimage=None, seed=0):
        data = {}
        attr = {}

        output = {}
        
        # TODO: Is this right?
        image_shape = dataset[self.input_dataset[0]].shape
        batch_indexes = np.zeros(self.batch_size, dtype=np.int16)
        
        if subimage is None or len(subimage) == 1:
            required_slices = range(image_shape[1])
        else:
            required_slices = range(image_shape[1])[subimage[1]]
            if isinstance(required_slices, int):
                required_slices = [required_slices]
        
        b = 0
        for coord_z in range(image_shape[1]):
            if coord_z not in required_slices:
                continue
            
            im = {x:dataset[x] for x in self.input_dataset}
            for x in self.input_dataset:
                if x not in data:
                    if is_numpy(im[x]):
                        data[x] = np.empty((self.batch_size, im[x].shape[0]) + (im[x].shape[2:]), dtype=im[x].dtype)
                    else:
                        data[x] = torch.empty((self.batch_size, im[x].shape[0]) + (im[x].shape[2:]), dtype=im[x].dtype, device=im[x].device)
                data[x][b] = im[x][:,coord_z]
        
            batch_indexes[b] = coord_z
            
            b += 1
            if b == self.batch_size:
                s = get_random_state()
                set_seed(seed)
                for processor in self.processors:
                    processor(data, attr)
                    
                out = self.model.apply(data)
                set_random_state(s)

                for i in range(b):
                    for x in out:
                        if x not in output:
                            if is_numpy(out):
                                output[x] = torch.empty((out[x].shape[1], image_shape[1]) + out[x].shape[2:], dtype=out[x].dtype)
                            else:
                                output[x] = torch.empty((out[x].shape[1], image_shape[1]) + out[x].shape[2:], dtype=out[x].dtype, device=out[x].device)
                        output[x][:,batch_indexes[i]] = out[x][i,:]

                b = 0
                data = {}
                attr = {}
        
        if b > 0:
            data = {x:data[x][0:b] for x in data}
            s = get_random_state()
            set_seed(seed)
            for processor in self.processors:
                processor(data, attr)

            out = self.model.apply(data)
            set_random_state(s)
            
            for i in range(b):
                for x in out:
                    if x not in output:
                        if is_numpy(out):
                            output[x] = torch.empty((out[x].shape[1], image_shape[1]) + out[x].shape[2:], dtype=out[x].dtype)
                        else:
                            output[x] = torch.empty((out[x].shape[1], image_shape[1]) + out[x].shape[2:], dtype=out[x].dtype, device=out[x].device)
                    output[x][:,batch_indexes[i]] = out[x][i,:]
            
        if subimage is not None:
            for x in output:
                output[x] = output[x][subimage]
        
        return output