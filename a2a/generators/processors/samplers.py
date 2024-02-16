import numpy as np
import torch

import random

from .processor import Processor
from a2a.generators.factory import processor
from a2a.processors import is_numpy
from a2a.utils.globals import get_global


@processor('ItemSampler')
class ItemSampler(Processor):
    def __init__(self, dataset='train', input_dataset=[], output_dataset=None, batch_size=1):
        super().__init__(dataset=input_dataset, output_dataset=output_dataset)
        
        self.data_dataset = dataset
        self.batch_size = batch_size

    
    def __call__(self, data, attrs):
        dataset = get_global('datasets')[self.data_dataset]
        
        # TODO: Populate attrs
        # TODO: Could return a view?
        for i in range(self.batch_size):
            im, _ = dataset.random_item()
            for j,o in zip(self.dataset, self.output_dataset):
                # if o not in data:
                if i == 0:
                    if is_numpy(im[j]):
                        data[o] = np.empty((self.batch_size,) + im[j].shape, dtype=im[j].dtype)
                    else:
                        data[o] = torch.empty((self.batch_size,) + im[j].shape, dtype=im[j].dtype, device=im[j].device)
                data[o][i] = im[j]

@processor('SliceSampler')
class SliceSampler(Processor):
    def __init__(self, dataset='train', input_dataset=[], output_dataset=None, batch_size=1):
        super().__init__(dataset=input_dataset, output_dataset=output_dataset)
        
        self.data_dataset = dataset
        self.batch_size = batch_size

    
    def __call__(self, data, attrs):
        dataset = get_global('datasets')[self.data_dataset]
        
        # TODO: Populate attrs
        # TODO: Could return a view?
        for i in range(self.batch_size):
            im, _ = dataset.random_item()
            index = random.randint(0, im[self.dataset[0]].shape[1] - 1)
            for j,o in zip(self.dataset, self.output_dataset):
                # if o not in data:
                if i == 0:
                    if is_numpy(im[j]):
                        data[o] = np.empty((self.batch_size, im[j].shape[0]) + (im[j].shape[2:]), dtype=im[j].dtype)
                    else:
                        data[o] = torch.empty((self.batch_size, im[j].shape[0]) + (im[j].shape[2:]), dtype=im[j].dtype, device=im[j].device)
                data[o][i] = im[j][:,index]


@processor('PixelSampler')
class PixelSampler(Processor):
    def __init__(self, dataset='train', input_dataset=[], output_dataset=None, batch_size=1):
        super().__init__(dataset=input_dataset, output_dataset=output_dataset)
        
        self.data_dataset = dataset
        self.batch_size = batch_size

    
    def __call__(self, data, attrs):
        dataset = get_global('datasets')[self.data_dataset]
        
        # TODO: Populate attrs
        
        # TODO: Number of images parameter?
        im, _ = dataset.random_item()
        n_ind = np.prod(im[self.dataset[0]].shape[1:])
        if is_numpy(im):
            inds = np.random.randint(n_ind,size=self.batch_size)
        else:
            inds = torch.randint(n_ind, size=self.batch_size, device=im.device)
        
        for j,o in zip(self.dataset, self.output_dataset):
            tmp = im[j].reshape(im[j].shape[0], -1)
            data[o] = tmp[:,inds].transpose(1,0)[:,:,None,None] # TODO: Hack to get image shape...


@processor('StaticSampler')
class StaticSampler(Processor):
    def __init__(self, dataset='train', input_dataset=[], output_dataset=None, index=[], slice_index=None):
        super().__init__(dataset=input_dataset, output_dataset=output_dataset)
        
        self.data_dataset = dataset
        self.index = index
        self.slice_index = slice_index

    def __call__(self, data, attrs):
        dataset = get_global('datasets')[self.data_dataset]
        
        for k, i in enumerate(self.index):
            im, _ = dataset.indexed_item(i)
            
            for j,o in zip(self.dataset, self.output_dataset):
                # if o not in data:
                if k == 0:
                    if self.slice_index is None:
                        shp = (len(self.index),) + im[j].shape
                    else:
                        shp = (len(self.index), im[j].shape[0]) + im[j].shape[2:]
                    if is_numpy(im[j]):
                        data[o] = np.empty(shp, dtype=im[j].dtype)
                    else:
                        data[o] = torch.empty(shp, dtype=im[j].dtype, device=im[j].device)
                if self.slice_index is None:
                    data[o][i] = im[j]
                else:
                    data[o][i] = im[j][:,self.slice_index[k]]


@processor('CroppedSliceSampler')
class CroppedSliceSampler(Processor):
    def __init__(self, dataset='train', input_dataset=[], output_dataset=None, batch_size=1, shape=(256,256)):
        super().__init__(dataset=input_dataset, output_dataset=output_dataset)
        
        self.data_dataset = dataset
        self.batch_size = batch_size
        self.shape = tuple(shape)

    
    def __call__(self, data, attrs):
        dataset = get_global('datasets')[self.data_dataset]
        
        # TODO: Populate attrs
        # TODO: Could return a view?
        for i in range(self.batch_size):
            im, _ = dataset.random_item()
            index = random.randint(0, im[self.dataset[0]].shape[1] - 1)
            cy = random.randint(0, im[self.dataset[0]].shape[2] - self.shape[0])
            cx = random.randint(0, im[self.dataset[0]].shape[3] - self.shape[1])
            for j,o in zip(self.dataset, self.output_dataset):
                if i == 0:
                    if is_numpy(im[j]):
                        data[o] = np.empty((self.batch_size, im[j].shape[0]) + self.shape, dtype=im[j].dtype)
                    else:
                        data[o] = torch.empty((self.batch_size, im[j].shape[0]) + self.shape, dtype=im[j].dtype, device=im[j].device)
                
                
                
                data[o][i] = im[j][:,index,cy:cy+self.shape[0],cx:cx+self.shape[1]]


@processor('DatasetSliceSampler')
class DatasetSliceSampler(Processor):
    def __init__(self, dataset='train', input_dataset=[], output_dataset=None, batch_size=1):
        super().__init__(dataset=input_dataset, output_dataset=output_dataset)
        
        self.data_dataset = dataset
        self.batch_size = batch_size
        
        self.current_index = 0
        self.current_slice = 0
        
        self.drop_last = False

    def __call__(self, data, attrs):
        dataset = get_global('datasets')[self.data_dataset]
        
        done = False
        while not done:
            inds = []
            for i in range(self.batch_size):
                im, _ = dataset.indexed_item(self.current_index) # TODO: Dynamic loader won't like this
                inds.append((self.current_index, self.current_slice))
                
                self.current_slice += 1
                if self.current_slice >= im[self.dataset[0]].shape[1]:
                    self.current_slice = 0
                    self.current_index += 1
                    if self.current_index >= dataset.number_of_items():
                        self.current_index = 0
                        break
            
            done = not (self.drop_last and len(inds) != self.batch_size)

        for i,(ind,s) in enumerate(inds):
            im, _ = dataset.indexed_item(ind)
            
            for j,o in zip(self.dataset, self.output_dataset):
                if i == 0:
                    if is_numpy(im[j]):
                        data[o] = np.empty((len(inds), im[j].shape[0]) + (im[j].shape[2:]), dtype=im[j].dtype)
                    else:
                        data[o] = torch.empty((len(inds), im[j].shape[0]) + (im[j].shape[2:]), dtype=im[j].dtype, device=im[j].device)
                data[o][i] = im[j][:,s]