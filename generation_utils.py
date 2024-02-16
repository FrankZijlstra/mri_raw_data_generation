from a2a.generators.processors.processor import Processor
from a2a.generators.factory import processor
from a2a.networks import network
from a2a.networks.utils import Reshape, initialize_weights

from torch import nn

@processor('MRIRawData')
class MRIRawData(Processor):
    def __init__(self, dataset=[], output_dataset=None, dataset_phase='', dataset_csm='', delete_phase=True, delete_csm=True):
        super().__init__(dataset=dataset, output_dataset=output_dataset)
        self.dataset_phase = dataset_phase
        self.dataset_csm = dataset_csm
        self.delete_phase = delete_phase
        self.delete_csm = delete_csm
        
    def __call__(self, data, attrs):
        im = data[self.dataset[0]]
        phase = data[self.dataset_phase]
        csm = data[self.dataset_csm]
        
        # im: B 1 YX (magnitude)
        # phase: B 2 YX (real/imag)
        # csm: B C*2 YX (real coils, imag coils)
        ncoils = csm.shape[1] // 2
        phase = phase[:,:1] + 1j * phase[:,1:]
        phase /= (abs(phase)+1e-12)

        data[self.output_dataset[0]] = (im * phase) * (csm[:,:ncoils] + 1j*csm[:,ncoils:])
        # out: B C YX complex
        
        if self.delete_phase:
            del data[self.dataset_phase]
        if self.delete_csm:
            del data[self.dataset_csm]

@network('FCN2DMLP', normalization_parameters=['normalization'], activation_parameters=['activation'])
class FCN2DMLP(nn.Module):
    def __init__(self, input_channels=1, hidden_channels_mlp=32, hidden_channels_fcn=32, output_channels=1, image_shape=[64,64], filter_size=(3,3), padding=(1,1), padding_mode='zeros', bias=None, activation=nn.SELU, normalization=None):
        super().__init__()
        
        if bias is None:
            bias = not normalization

        layerlist = [nn.Conv2d(input_channels, hidden_channels_fcn, kernel_size=7, padding='same', padding_mode=padding_mode),
                     activation(),
                     nn.Conv2d(hidden_channels_fcn, hidden_channels_fcn, kernel_size=4, padding=1, stride=2, padding_mode=padding_mode)]
        if normalization:
            layerlist += [normalization(hidden_channels_fcn)]
        
        for i in range(3):
            layerlist += [activation(),
                         nn.Conv2d(hidden_channels_fcn, hidden_channels_fcn, kernel_size=7, padding='same', padding_mode=padding_mode)]
            if normalization:
                layerlist += [normalization(hidden_channels_fcn)]
            layerlist += [activation(),
                         nn.Conv2d(hidden_channels_fcn, hidden_channels_fcn, kernel_size=4, padding=1, stride=2, padding_mode=padding_mode)]
            if normalization:
                layerlist += [normalization(hidden_channels_fcn)]

        layerlist += [activation(),
                     Reshape(shape=[-1]),
                     nn.Linear(image_shape[0]//16*image_shape[1]//16*hidden_channels_fcn, hidden_channels_mlp*2),
                     activation(),
                     nn.Linear(hidden_channels_mlp*2, hidden_channels_mlp),
                     activation(),
                     nn.Linear(hidden_channels_mlp,output_channels)]
 
        self.model = nn.Sequential(*layerlist)
        self.model.apply(initialize_weights)
            
    def forward(self, x):
        return self.model(x)
