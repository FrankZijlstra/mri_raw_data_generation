from . import preprocessor
from .preprocessor_base import Preprocessor, register_op_preprocessor
from a2a.processors.misc import Multiply, Scale, ComplexToChannels, ChannelsToComplex, KeepComplexPhase, SplitChannels, Magnitude, GenerateNoise
from a2a.processors.cropping import CropChannels

register_op_preprocessor('Multiply', Multiply)
register_op_preprocessor('Scale', Scale)
register_op_preprocessor('ComplexToChannels', ComplexToChannels, channel_dim=0)
register_op_preprocessor('ChannelsToComplex', ChannelsToComplex, channel_dim=0)
register_op_preprocessor('KeepComplexPhase', KeepComplexPhase)
register_op_preprocessor('SplitChannels', SplitChannels, channel_dim=0)
register_op_preprocessor('Magnitude', Magnitude)
register_op_preprocessor('CropChannels', CropChannels, channel_dim=0)
register_op_preprocessor('GenerateNoise', GenerateNoise)

from a2a.processors.fourier import FFT2D, IFFT2D
register_op_preprocessor('FFT2D', FFT2D)
register_op_preprocessor('IFFT2D', IFFT2D)

@preprocessor('Remove')
class Remove(Preprocessor):
    def process(self, dataset, attrs):
        for x in self.dataset:
            del dataset[x]
            del attrs[x]
        return None
    def process_inverse(self, dataset, attrs, inverse_parameters):
        pass
