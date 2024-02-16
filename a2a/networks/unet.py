import torch
import torch.nn as nn

from .utils import initialize_weights
from . import network

from a2a.processors.cropping import Pad2D, Crop2D

@network('UNet2D', normalization_parameters=['normalization'], activation_parameters=['activation'])
class UNet2D(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, hidden_channels=32, levels=4, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', bias=None, activation=nn.ReLU, normalization=None, dropout=None, prepad=False):
        super().__init__()
        self.levels = levels
        
        if bias is None:
            bias = not normalization
        
        self.prepad = prepad
        
        layers = [nn.Conv2d(input_channels, hidden_channels, (1, 1)),
                  activation()]
        
        ch = int(hidden_channels * 2 ** (levels-1))
        block = None
        for level in range(levels):
            # TODO: input_channels is bit messy
            block = UNet2DBlock(int(ch/2) if level<levels-1 else ch, ch, inner_block=block, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=bias, activation=activation, normalization=normalization, dropout=dropout)
            ch = int(ch / 2)

        layers += [block,
                   nn.Conv2d(hidden_channels, output_channels, (1, 1))]
        
        self.model = nn.Sequential(*layers)
        self.model.apply(initialize_weights)

    def forward(self, x):
        if self.prepad:
            multiple = 2**(self.levels-1)
            input_shape = x.shape[2:]
            r = x.shape[2] % multiple
            if r == 0:
                size_y = x.shape[2]
            else:
                size_y = x.shape[2] + multiple - r
            r = x.shape[3] % multiple
            if r == 0:
                size_x = x.shape[3]
            else:
                size_x = x.shape[3] + multiple - r
            x = Pad2D(size_y, size_x)(x)
            x = self.model(x)
            return Crop2D(size_y=input_shape[0], size_x=input_shape[1])(x)
        else:
            return self.model(x)

class UNet2DBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, inner_block=None, kernel_size=(3,3), padding=(1,1), padding_mode='zeros', bias=None, activation=nn.ReLU, normalization=None, dropout=None):
        super().__init__()
        if bias is None:
            bias = not normalization
        
        self.hidden_channels = hidden_channels
        
        layers1 = [nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias)]
        if normalization:
            layers1 += [normalization(hidden_channels)]
        layers1 += [activation()]
        if dropout:
            layers1 += [nn.Dropout2d(p=dropout)]

        layers1 += [nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias)]
        if normalization:
            layers1 += [normalization(hidden_channels)]
        layers1 += [activation()]
        if dropout:
            layers1 += [nn.Dropout2d(p=dropout)]
        
        inner_block_channels = 0
        if inner_block:
            layers_inner = [nn.AvgPool2d((2,2)),
                      inner_block,
                      nn.Upsample(scale_factor=2)]
            
            self.inner_model = nn.Sequential(*layers_inner)
            
            inner_block_channels = inner_block.hidden_channels
        else:
            self.inner_model = None
            
        layers2 = [nn.ConvTranspose2d(hidden_channels + inner_block_channels, hidden_channels, kernel_size, padding=padding, bias=bias)]
        if normalization:
            layers2 += [normalization(hidden_channels)]
        layers2 += [activation()]
        if dropout:
            layers2 += [nn.Dropout2d(p=dropout)]
            
        layers2 += [nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)]
        if normalization:
            layers2 += [normalization(hidden_channels)]
        layers2 += [activation()]
        if dropout:
            layers2 += [nn.Dropout2d(p=dropout)]
            
        self.model1 = nn.Sequential(*layers1)
        self.model2 = nn.Sequential(*layers2)

    def forward(self, x):
        x = self.model1(x)

        if self.inner_model:
            skip = x            
            x = self.inner_model(x)
            x = torch.cat([x, skip], dim=1)
               
        x = self.model2(x)
        return x
