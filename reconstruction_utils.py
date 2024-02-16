import torch
import numpy as np

from a2a.networks import network
from a2a.processors.fourier import FFT2D, IFFT2D

import copy

from a2a.networks.utils import initialize_weights
from a2a.processors.processor import UnaryProcessor
from a2a.preprocessors.preprocessor_base import register_op_preprocessor
from a2a.generators.processors.processor import register_op_processor

class RSSRecon(UnaryProcessor):
    def __init__(self, channel_dim=1):
        super().__init__()
        self.channel_dim = channel_dim
        
    def call_numpy(self, image, allow_in_place=False):
        return np.sqrt((abs(image)**2).sum(axis=self.channel_dim, keepdims=True))

    def call_torch(self, image, allow_in_place=False):
        return torch.sqrt((abs(image)**2).sum(dim=self.channel_dim, keepdims=True))

register_op_preprocessor('RSSRecon', RSSRecon, channel_dim=0)
register_op_processor('RSSRecon', RSSRecon)


@network('VarNet', network_parameters=['network', 'network_sme'])
class VarNet(torch.nn.Module):
    def __init__(self, network, network_sme, cascades=12, center_mask=[12,12]):
        super().__init__()
        
        # TODO: How to get [network for i in range(cascades)] ?
        # torch.nn.ModuleList(copy.deepcopy(network) for i in range(cascades)) ?
        self.network = torch.nn.ModuleList(copy.deepcopy(network) for i in range(cascades))
        for net in self.network:
            net.apply(initialize_weights)
        
        # self.network = network
        self.network_sme = network_sme
        self.cascades = cascades
        self.dc_weight = torch.nn.Parameter(torch.ones(1))
        # self.dc_weight = 1
        self.center_mask = center_mask
        
        self.ft = FFT2D()
        self.ift = IFFT2D()
        self.rss = RSSRecon()
    
    
    def sme(self, k_masked):
        b, c, h, w = k_masked.shape
        ims = self.ift(k_masked).view(b*c, 1, h, w)
        
        est_coils = self.network_sme(torch.cat((ims.real, ims.imag), dim=1))
        est_coils = (est_coils[:,:1] + 1j*est_coils[:,1:]).view(b, c, h, w) # TODO: could combine with RSS?
        
        # Root sum of squares
        rss = self.rss(est_coils)
        
        # Return estimated CSM
        return est_coils / rss
    
    def cascade(self, k_current, k_ref, mask, csm, i):
        soft_dc = (k_current - k_ref) * mask * self.dc_weight
        
        network_input = (self.ift(k_current) * csm.conj()).sum(dim=1, keepdims=True)
        
        m = network_input.mean(dim=[1,2,3], keepdims=True)
        s = network_input.std(dim=[1,2,3], keepdims=True) # Std over complex? Same as R.std() + 1j * I.std()?
        
        network_input = (network_input - m)/s
        
        model_term = self.network[i](torch.cat((network_input.real, network_input.imag), dim=1))
        model_term = model_term[:,:1] + 1j * model_term[:,1:]
        
        model_term = model_term*s + m
        
        model_term = self.ft(model_term * csm)
        
        return k_current - soft_dc - model_term
        
    
    def forward(self, k_ref):       
        mask = abs(k_ref[:,[0]]) > 1e-6 # TODO: Use actual mask
        center_mask = mask*0
        sy,sx = k_ref.shape[2:]
        ry = self.center_mask[0]
        rx = self.center_mask[1]
        
        pad_y = (sy - ry + 1) // 2
        pad_x = (sx - rx + 1) // 2
        
        # center_mask[:,:,sy//2-ry:sy//2+ry,sx//2-rx:sx//2+rx] = 1
        center_mask[:,:,pad_y:pad_y+ry,pad_x:pad_x+rx] = 1        
        
        csm = self.sme(k_ref * center_mask)
        
        # Iterate
        k_current = k_ref.clone()
        for i in range(self.cascades):
            k_current = self.cascade(k_current, k_ref, mask, csm, i)
        
        # Return RSS
        return self.rss(self.ift(k_current))

    
from a2a.generators.processors.processor import Processor, processor
import random
@processor('GenerateEquispacedMask')
class GenerateEquispacedMask(Processor):
    def __init__(self, output_dataset=None, batch_size=1, shape=[1,1], acceleration=4, center_fraction=0.08):
        super().__init__(dataset='', output_dataset=output_dataset)
        
        self.batch_size = batch_size
        self.shape = shape
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        
    def __call__(self, data, attrs):
        if type(self.batch_size) is str:
            batch_size = data[self.batch_size].shape[0]
        else:
            batch_size = self.batch_size
            
        if type(self.shape) is str:
            shape = list(data[self.shape].shape[2:])
        else:
            shape = list(self.shape)

        n_cols = shape[0]
        n_low = round(n_cols * self.center_fraction)
        mask = torch.zeros((batch_size,1) + tuple(shape), device='cuda:0')
        pad = (n_cols - n_low + 1) // 2
        mask[:,:,pad:pad+n_low,:] = 1
        
        for i in range(batch_size):
            offset = random.randint(0, round(self.acceleration)-1)
            # print(i,offset)
            mask[i,:,offset::self.acceleration,:] = 1
        
        data[self.output_dataset[0]] = mask

from a2a.preprocessors import preprocessor
from a2a.preprocessors.preprocessor_base import Preprocessor

from a2a.utils.utils import set_seed, get_random_state, set_random_state

@preprocessor('GenerateEquispacedMask')
class GenerateEquispacedMask(Preprocessor):
    def __init__(self, output_dataset=None, shape=[1,1], acceleration=4, center_fraction=0.08, seed=0):
        super().__init__(dataset='', output_dataset=output_dataset)
        
        self.shape = shape
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.seed = seed
        
    def process(self, data, attrs):
        if self.seed is not None:
            prev_state = get_random_state()
            set_seed(self.seed)
            
        if type(self.shape) is str:
            shape = list(data[self.shape].shape[1:])
        else:
            shape = list(self.shape)

        n_cols = shape[1]
        n_low = round(n_cols * self.center_fraction)
        mask = torch.zeros((1,) + tuple(shape), device='cuda:0')
        pad = (n_cols - n_low + 1) // 2
        mask[:,:,pad:pad+n_low,:] = 1
        
        for i in range(shape[0]):
            offset = random.randint(0, round(self.acceleration)-1)
            # print(i,offset)
            mask[0,i,offset::self.acceleration,:] = 1

        data[self.output_dataset[0]] = mask
        attrs[self.output_dataset[0]] = {}
        
        if self.seed is not None:
            set_random_state(prev_state)
        return None



@preprocessor('CalculateNoiseCovariance')
class CalculateNoiseCovariance(Preprocessor):
    def __init__(self, dataset=None, output_dataset=None):
        super().__init__(dataset=dataset, output_dataset=output_dataset)
        self.per_slice = True
        
    def process(self, data, attrs):
        # TODO: Mask dataset
        # TODO: Per-slice (provide ndim?)
        im = data[self.dataset[0]]
        
        # TODO: Determine complex or not
        if self.per_slice:
            out = np.empty((1,im.shape[1],im.shape[0],im.shape[0]), dtype=np.complex64)
            for i in range(im.shape[1]):
                # TODO: /2 only if complex
                out[:,i] = np.cov(im[:,i,:20,:20].reshape(im.shape[0],-1)).astype(np.complex64)
            
            data[self.output_dataset[0]] = out 
        else:
            data[self.output_dataset[0]] = np.cov(im[:,:,:20,:20].reshape(im.shape[0],-1)).astype(np.complex64)
        attrs[self.output_dataset[0]] = {}

        return None
