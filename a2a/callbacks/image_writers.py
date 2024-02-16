from . import callback
from .callback_base import Callback

import numpy as np
import torch

import matplotlib.pyplot as plt
from pathlib import Path

from a2a.utils.globals import get_global

def tensor2rgb(im, vmin=None, vmax=None):
    if not vmin:
        vmin = im.min()
    if not vmax:
        vmax = im.max()

    # im is c,y,x
    if im.shape[0] == 1:
        im = np.pad(im.transpose((2,1,0)), ((0,0), (0,0), (0, 3-im.shape[0])), mode='edge')
    elif im.shape[0] <= 3:
        im = np.pad(im.transpose((2,1,0)), ((0,0), (0,0), (0, 3-im.shape[0])), mode='constant', constant_values=vmin)
    else:
        im = im.transpose((2,1,0))[:,:,0:3]

    im = np.clip((im - vmin) / (vmax - vmin), 0, 1)
    return im

def tensor2grid(im, vmin=None, vmax=None, complex=True):
    if vmin is None:
        if complex:
            vmin = abs(im).min()
        else:
            vmin = im.min()
    if vmax is None:
        if complex:
            vmax = abs(im).max()
        else:
            vmax = im.max()

    nc = im.shape[0]
    if complex:
        nc = int(nc/2)
    
    s = int(np.ceil(np.sqrt(nc)))
    
    out = np.zeros((s * im.shape[1], s * im.shape[2], 3), dtype=im.dtype)
    cmap = plt.get_cmap('hsv')
    for i in range(s):
        for j in range(s):
            ind = i*s + j
            if ind >= nc:
                continue
            
            if complex:
                # out[i*im.shape[1]:(i+1)*im.shape[1], j*im.shape[2]:(j+1)*im.shape[2],:] = np.stack((im[ind],im[nc+ind],np.zeros(im[ind].shape)), axis=2)
                tmp = im[ind] + 1j * im[nc + ind]
                out[i*im.shape[1]:(i+1)*im.shape[1], j*im.shape[2]:(j+1)*im.shape[2],:] = cmap((np.angle(tmp) + np.pi)/(2*np.pi))[...,0:3] * abs(tmp)[...,None]
            else:
                out[i*im.shape[1]:(i+1)*im.shape[1], j*im.shape[2]:(j+1)*im.shape[2],:] = np.stack((im[ind],im[ind],im[ind]), axis=2)
    
    out = np.clip((out - vmin) / (vmax - vmin), 0, 1)
    return out


# Segmentation is one or more channels with values between 0 and 1, cmap needs to be categorical
def segmentation2rgb(im, cmap_name='tab10'):
    num_channels = im.shape[0]
    cmap = plt.get_cmap(cmap_name)

    # colours = cmap(range(num_channels))
    colours = cmap(np.linspace(0,1,num_channels))
    return np.clip(im.transpose((2,1,0)) @ colours[:,0:3], 0, 1)



@callback('OutputImageWriter')
class OutputImageWriter(Callback):
    def __init__ (self, filename_prefix='', model_runner='default', output_dataset='', grid=False, complex=True, segmentation=False):
        self.log_directory = get_global('log_directory')
        self.filename_prefix = filename_prefix
        self.model_runner = get_global('model_runners')[model_runner]
        self.segmentation = segmentation
        self.grid = grid
        self.output_dataset = output_dataset
        self.complex = complex
        

    def on_epoch_end (self, epoch_dict):
        output = self.model_runner({}, {})[self.output_dataset].cpu().numpy()

        if output.ndim == 2:
            output = output[np.newaxis]

        if self.segmentation:
            image = segmentation2rgb(output)
        elif self.grid:
            image = tensor2grid(output, vmin=output.min(), vmax=output.max(), complex=self.complex)
        else:
            image = tensor2rgb(output, vmin=output.min(), vmax=output.max())

        plt.imsave(self.log_directory / (self.filename_prefix + '_{}.png'.format(epoch_dict['epoch']+1)), image)

@callback('ImageWriter')
class ImageWriter(Callback):
    def __init__ (self, filename_prefix='', model_runner='default', dataset='validation', output_dataset='', dataset_index=0, grid=False, complex=True, segmentation=False, slice=8):
        self.log_directory = get_global('log_directory')
        self.filename_prefix = filename_prefix
        self.model_runner = get_global('model_runners')[model_runner]
        self.segmentation = segmentation
        self.grid = grid
        self.output_dataset = output_dataset
        self.complex = complex

        self.dataset = get_global('datasets')[dataset]
        self.dataset_index = dataset_index
        self.slice = slice

        self.image = get_global('datasets')[dataset].indexed_item(dataset_index)
        

    def on_epoch_end (self, epoch_dict):
        output = self.model_runner(self.image[0], self.image[1], subimage=(slice(None), self.slice))[self.output_dataset].cpu().numpy()

        if output.ndim == 2:
            output = output[np.newaxis]

        if self.segmentation:
            image = segmentation2rgb(output)
        elif self.grid:
            image = tensor2grid(output, vmin=output.min(), vmax=output.max(), complex=self.complex)
        else:
            image = tensor2rgb(output, vmin=output.min(), vmax=output.max())

        plt.imsave(self.log_directory / (self.filename_prefix + '_{}.png'.format(epoch_dict['epoch']+1)), image)


@callback('PairedImageWriter', preprocessor_parameters=['output_processors'])
class PairedImageWriter(Callback):
    def __init__ (self, filename_prefix='', model_runner='default', dataset='validation', input_dataset=None, output_dataset='', dataset_index=0, grid=False, complex=False, segmentation=False, slice=8, output_processors=[]):
        self.log_directory = get_global('log_directory')
        self.filename_prefix = filename_prefix
        self.model_runner = get_global('model_runners')[model_runner]
        self.segmentation = segmentation
        self.grid = grid
        self.complex = complex
        self.output_dataset = output_dataset


        self.dataset = get_global('datasets')[dataset]
        self.dataset_index = dataset_index
        self.slice = slice

        self.image = get_global('datasets')[dataset].indexed_item(dataset_index)
        
        if input_dataset is None:
            input_dataset = [output_dataset]
        
        data = {x:self.image[0][x] for x in input_dataset}
        attr = {x:self.image[1][x] for x in input_dataset}
        
        for processor in output_processors:
            processor.process(data, attr)
        
        self.true_output = data[self.output_dataset][:,slice]

    
    def on_training_start(self, epoch_dict):
        true_output = self.true_output
        image = true_output
        if self.segmentation:
            image = segmentation2rgb(image)
        elif self.grid:
            image = tensor2grid(image, vmin=self.true_output.min(), vmax=self.true_output.max(), complex=self.complex)
        else:
            image = tensor2rgb(image, vmin=self.true_output.min(), vmax=self.true_output.max())
        plt.imsave(self.log_directory / (self.filename_prefix + '_true.png'), image)  

    def on_epoch_end (self, epoch_dict):
        output = self.model_runner(self.image[0], self.image[1], subimage=(slice(None), self.slice))[self.output_dataset].cpu().numpy()
        true_output = self.true_output

        if output.ndim == 2:
            output = output[np.newaxis]
            true_output = true_output[np.newaxis]

        if self.segmentation:
            image = segmentation2rgb(output)
        elif self.grid:
            image = tensor2grid(output, vmin=self.true_output.min(), vmax=self.true_output.max(), complex=self.complex)
        else:
            image = tensor2rgb(output, vmin=self.true_output.min(), vmax=self.true_output.max())

        plt.imsave(self.log_directory / (self.filename_prefix + '_{}.png'.format(epoch_dict['epoch']+1)), image)


# TODO: Have a look at seeded generation, how to deal with this?

@callback('SeededPairedImageWriter', preprocessor_parameters=['output_processors'])
class SeededPairedImageWriter(Callback):
    def __init__ (self, filename_prefix='', model_runner='default', dataset='validation', input_dataset=None, output_dataset='', dataset_index=0, shape=[1,1], segmentation=False, grid=False, complex=True, slice=8, output_processors=[]):
        self.log_directory = get_global('log_directory')
        self.filename_prefix = filename_prefix
        self.model_runner = get_global('model_runners')[model_runner]
        self.segmentation = segmentation
        self.grid = grid
        self.complex = complex
        self.output_dataset = output_dataset

        self.shape = shape

        self.dataset = get_global('datasets')[dataset]
        self.dataset_index = dataset_index
        self.slice = slice

        self.image = get_global('datasets')[dataset].indexed_item(dataset_index)

        if input_dataset is None:
            input_dataset = [output_dataset]

        data = {x:self.image[0][x] for x in input_dataset}
        attr = {x:self.image[1][x] for x in input_dataset}

        for processor in output_processors:
            processor.process(data, attr)
        
        self.true_output = data[self.output_dataset][:,slice]

    
    def on_training_start(self, epoch_dict):
        true_output = self.true_output
        image = true_output
        if self.segmentation:
            image = segmentation2rgb(image)
        elif self.grid:
            if self.complex:
                vmin = 0
                vmax = abs(self.true_output[0] + 1j * self.true_output[1]).max()
            else:
                vmin = self.true_output.min()
                vmax = self.true_output.max()
            image = tensor2grid(image, vmin=vmin, vmax=vmax, complex=self.complex)
            # image = tensor2grid(image, complex=self.complex)
        else:
            image = tensor2rgb(image, vmin=self.true_output.min(), vmax=self.true_output.max())
        plt.imsave(self.log_directory / (self.filename_prefix + '_true.png'), image)


    def on_epoch_end (self, epoch_dict):
        image = None
        for i in range(self.shape[0]):
            image1 = None
            for j in range(self.shape[1]):
                output = self.model_runner(self.image[0], self.image[1], subimage=(slice(None), self.slice), seed=i*self.shape[1] + j)[self.output_dataset].cpu().numpy()
                if image1 is None:
                    image1 = output
                else:
                    image1 = np.concatenate((image1, output), axis=1)
            if image is None:
                image = image1
            else:
                image = np.concatenate((image, image1), axis=2)

        if self.segmentation:
            image = segmentation2rgb(image)
        elif self.grid:
            if self.complex:
                vmin = 0
                vmax = abs(self.true_output[0] + 1j * self.true_output[1]).max()
            else:
                vmin = self.true_output.min()
                vmax = self.true_output.max()
            image = tensor2grid(image, vmin=vmin, vmax=vmax, complex=self.complex)
            # image = tensor2grid(image, vmin=self.true_output.min(), vmax=self.true_output.max(), complex=self.complex)
            # image = tensor2grid(image, complex=self.complex)
        else:
            image = tensor2rgb(image, vmin=self.true_output.min(), vmax=self.true_output.max())

        plt.imsave(self.log_directory / (self.filename_prefix + '_{}.png'.format(epoch_dict['epoch']+1)), image)

