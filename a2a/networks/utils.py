import torch
import torch.nn as nn
import torch.nn.init as init

import math

from . import network

def he_normal(tensor, gain=1):
    fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in))
    with torch.no_grad():
        return tensor.normal_(0, std)

def initialize_weights(layer):
    if type(layer) == nn.Conv2d:
        he_normal(layer.weight)
    if type(layer) == nn.Conv3d:
        he_normal(layer.weight)
    if type(layer) == nn.ConvTranspose2d:
        he_normal(layer.weight)
    if type(layer) == nn.ConvTranspose3d:
        he_normal(layer.weight)

@network('Identity')
class Identity(nn.Module):    
    def forward(self, x):
        return x

@network('Reshape')
class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)
    def forward(self, x):
        return x.reshape((x.shape[0],) + self.shape)

@network('Constant')
class Constant(nn.Module):
    def __init__(self, image_shape):
        super().__init__()
        self.image_shape = tuple(image_shape)
    def forward(self, x):
        return x.reshape(x.shape + (1,) * len(self.image_shape)) * torch.ones(x.shape[0:2] + self.image_shape, dtype=x.dtype, device=x.device)

@network('Concatenate', network_parameters=['network'])
class Concatenate(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
    def forward(self, *x):
        return self.network(torch.cat(x, dim=1))
