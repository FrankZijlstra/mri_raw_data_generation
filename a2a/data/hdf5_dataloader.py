import h5py
import numpy as np

from pathlib import Path

from a2a.utils.factory import Factory
from . import dataloader
from .dataloader_base import DataLoader

hdf5channel_processor = Factory('HDF5 channel processor')

class HDF5ChannelProcessor:
    def __call__(self, hdf5_file):
        pass

@hdf5channel_processor('CpxPhase')
class CpxPhaseHDF5ChannelProcessor(HDF5ChannelProcessor):
    def __init__(self, channel):
        self.channel = channel
    
    def __call__(self, hdf5_file):
        return np.angle(np.array(hdf5_file[self.channel])), dict(hdf5_file[self.channel].attrs.items())


@hdf5channel_processor('CpxPhaseRI')
class CpxPhaseRIHDF5ChannelProcessor(HDF5ChannelProcessor):
    def __init__(self, channel):
        self.channel = channel
    
    def __call__(self, hdf5_file):
        tmp = np.exp(1j * np.angle(np.array(hdf5_file[self.channel])))
        return np.stack((tmp.real, tmp.imag), axis=0), dict(hdf5_file[self.channel].attrs.items())
    

@hdf5channel_processor('CpxMagnitude')
class CpxMagnitudeHDF5ChannelProcessor(HDF5ChannelProcessor):
    def __init__(self, channel):
        self.channel = channel
    
    def __call__(self, hdf5_file):
        return abs(np.array(hdf5_file[self.channel])), dict(hdf5_file[self.channel].attrs.items())


@hdf5channel_processor('Phase')
class PhaseHDF5ChannelProcessor(HDF5ChannelProcessor):
    def __init__(self, real_channel, imag_channel):
        self.real_channel = real_channel
        self.imag_channel = imag_channel
    
    def __call__(self, hdf5_file):
        return np.array(hdf5_file[self.real_channel]) + 1j * np.array(hdf5_file[self.imag_channel]), dict(hdf5_file[self.real_channel].attrs.items())

@hdf5channel_processor('ComplexToChannels')
class ComplexToChannelsHDF5ChannelProcessor(HDF5ChannelProcessor):
    def __init__(self, channel, mode='concatenate'):
        self.channel = channel
        self.mode = mode
        # TODO: More control over how it concatenates/stacks
    
    def __call__(self, hdf5_file):
        tmp = np.array(hdf5_file[self.channel])
        if self.mode == 'stack':
            return np.stack((tmp.real, tmp.imag), axis=0), dict(hdf5_file[self.channel].attrs.items())
        else:
            return np.concatenate((tmp.real, tmp.imag), axis=0), dict(hdf5_file[self.channel].attrs.items())


@dataloader('HDF5')
class HDF5DataLoader(DataLoader):
    def __init__(self, data_dir, channels, dtype=None):
        self.data_dir = data_dir
        self.channels = channels
        if dtype:
            self.dtype = {'float32':np.float32, 'float64':np.float64, 'complex64':np.complex64, 'complex128':np.complex128}[dtype]
        else:
            self.dtype = None
    
    def get_valid_indices(self):
        return [x.name[1:-3] for x in Path(self.data_dir).glob('P*.h5')]
    
    def get_num_channels(self):
        return len(self.channels)
    
    def load(self, index):
        print('Loading', index, self.channels)
        f = h5py.File(Path(self.data_dir) / f"P{index}.h5", "r")
        
        ds = []
        attrs = []
        c = 0
        shape = None
        dtype = self.dtype
        
        for i, channel in enumerate(self.channels):
            if isinstance(channel, dict):
                processor = hdf5channel_processor.create(channel)
                d, a = processor(f)
            else:
                d = np.array(f[channel])
                a = dict(f[channel].attrs.items())
            a['index'] = index
                
            if d.ndim <= 3:
                c += 1
                
                if not shape:
                    shape = d.shape
                assert(shape == d.shape)
            elif d.ndim == 4: # Assume 4D data is CZYX
                c += d.shape[0]
                
                if not shape:
                    shape = d.shape[1:]
                assert(shape == d.shape[1:])
            
            if self.dtype:
                if not np.can_cast(d.dtype, self.dtype, 'same_kind'):
                    raise Exception(f'Cannot cast {d.dtype} to {self.dtype}')
            elif dtype:
                # TODO: Does not assure that all previous d's can be cast to the new type...
                if not np.can_cast(d.dtype, dtype):
                    if np.can_cast(dtype, d.dtype):
                        dtype = d.dtype
                    else:
                        raise Exception(f'Cannot cast {d.dtype} to {dtype}')
            else:
                dtype = d.dtype

            ds.append(d)
            attrs.append(a)
        
        if self.dtype:
            dtype = self.dtype

        data = np.empty((c,) + shape, dtype=dtype)
        print(shape)
        c = 0
        for d in ds:
            if d.ndim <= 3:
                data[c] = d
                c += 1
            elif d.ndim == 4:
                data[c:c+d.shape[0]] = d
                c += d.shape[0]
        
        data.flags.writeable = False
        
        return data, attrs
