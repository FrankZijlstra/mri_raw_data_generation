import torch
import torch.nn as nn

from . import network
from .utils import initialize_weights


@network('MLP', normalization_parameters=['normalization'], activation_parameters=['activation'])
class MLP(nn.Module):
    def __init__(self, input_shape=[1], hidden_channels=[], output_shape=[1], activation=nn.SELU):
        super().__init__()
        self.output_shape = tuple(output_shape)
        
        c_out = 1
        for x in self.output_shape:
            c_out *= x
        
        c = 1
        for x in input_shape:
            c *= x
        
        layerlist = []
        
        for h in hidden_channels:
            layerlist += [nn.Linear(c, h)]
            c = h
            layerlist += [activation()]

        layerlist += [nn.Linear(c, c_out)]

        self.model = nn.Sequential(*layerlist)
        self.model.apply(initialize_weights)
            
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.model(x)
        x = x.reshape((x.shape[0],) + self.output_shape)
        return x

