import torch
import torch.nn as nn

from . import network

from .utils import initialize_weights

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return x.view((x.shape[0],) + self.shape)

# TODO: Merge into base FCN with changeable convolution layer (nn.Conv2d or nn.Conv3d)
@network('FCN2D', normalization_parameters=['normalization'], activation_parameters=['activation'])
class FCN2D(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=32, output_channels=1, layers=3, filter_size=(3,3), padding=(1,1), padding_mode='zeros', bias=None, activation=nn.ReLU, normalization=None, use_sigmoid=False, reshape=None):
        super().__init__()
        
        if bias is None:
            bias = not normalization
        
        layerlist = [nn.Conv2d(input_channels, hidden_channels, kernel_size=(1,1)),
                activation()]

        for l in range(layers):
            layerlist += [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=filter_size, padding=padding, padding_mode=padding_mode, bias=bias)]
            if normalization:
                layerlist += [normalization(hidden_channels)]
            layerlist += [activation()]
        
        layerlist += [nn.Conv2d(hidden_channels, output_channels, kernel_size=(1,1))]
        
        if use_sigmoid:
            layerlist += [nn.Sigmoid()]
        
        if reshape is not None:
            layerlist += [Reshape(reshape)]
        
        self.model = nn.Sequential(*layerlist)
        self.model.apply(initialize_weights)
            
    def forward(self, x):
        return self.model(x)

@network('FCN3D', normalization_parameters=['normalization'], activation_parameters=['activation'])
class FCN3D(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=32, output_channels=1, layers=3, filter_size=(3,3,3), padding=(1,1,1), padding_mode='zeros', bias=None, activation=nn.ReLU, normalization=None, use_sigmoid=False):
        super().__init__()
        
        if bias is None:
            bias = not normalization
            
        layerlist = [nn.Conv3d(input_channels, hidden_channels, kernel_size=(1,1,1)),
                activation()]
        
        for l in range(layers):
            layerlist += [nn.Conv3d(hidden_channels, hidden_channels, kernel_size=filter_size, padding=padding, padding_mode=padding_mode, bias=bias)]
            if normalization:
                layerlist += [normalization(hidden_channels)]
            layerlist += [activation()]
        
        layerlist += [nn.Conv3d(hidden_channels, output_channels, kernel_size=(1,1,1))]
        
        if use_sigmoid:
            layerlist += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*layerlist)
        self.model.apply(initialize_weights)
            
    def forward(self, x):
        return self.model(x)
