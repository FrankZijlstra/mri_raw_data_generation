import torch
import numpy as np

from .processor import NullaryProcessor, UnaryProcessor, BinaryProcessor, convert_dtype_torch, convert_dtype_numpy, is_complex_dtype

class Add(BinaryProcessor):
    def __init__(self):
        super().__init__()
    
    def __call__(self, image1, image2, allow_in_place=False):
        if allow_in_place:
            image1 += image2
            return image1
        else:
            return image1 + image2

class Subtract(BinaryProcessor):
    def __init__(self):
        super().__init__()
    
    def __call__(self, image1, image2, allow_in_place=False):
        if allow_in_place:
            image1 -= image2
            return image1
        else:
            return image1 - image2

class Multiply(BinaryProcessor):
    def __init__(self):
        super().__init__()
    
    def __call__(self, image1, image2, allow_in_place=False):
        if allow_in_place:
            image1 *= image2
            return image1
        else:
            return image1 * image2

class Divide(BinaryProcessor):
    def __init__(self):
        super().__init__()
    
    def __call__(self, image1, image2, allow_in_place=False):
        if allow_in_place:
            image1 /= image2
            return image1
        else:
            return image1 / image2


class Scale(UnaryProcessor):
    def __init__(self, scale=1, offset=0):
        super().__init__()
        self.scale = scale
        self.offset = offset
    
    def __call__(self, image, allow_in_place=False):
        if allow_in_place:
            image *= self.scale
            image += self.offset
            return image
        else:
            return image * self.scale + self.offset

class PixelShift2D(UnaryProcessor):
    def __init__(self, shift_y=0, shift_x=0, circular=True, pad_value=0):
        super().__init__()
        self.shift_y = shift_y
        self.shift_x = shift_x
        self.circular = circular # TODO: What function to use if False?
    
    def call_numpy(self, image, allow_in_place=False):
        return np.roll(image, [self.shift_y, self.shift_x], axis=[-2,-1])
    def call_torch(self, image, allow_in_place=False):
        return torch.roll(image, [self.shift_y, self.shift_x], dims=[-2,-1])


class TransposeFlip(UnaryProcessor):
    def __init__(self, transpose=None, flip=None):
        super().__init__()
        self.transpose = transpose
        self.flip = flip
        
        self.ndim = 0
        if transpose is not None:
            self.ndim = len(transpose)
        # TODO: this is not correct, flip is list of flipped dimensions, not list of all dimensions
        #if flip is not None:
        #    self.ndim = max(self.ndim, len(flip))
        
    def call_numpy(self, image, allow_in_place=False):
        if self.transpose is not None:
            # Creates view
            image = image.transpose(tuple(range(image.ndim - self.ndim)) + tuple(x + image.ndim - self.ndim for x in self.transpose))
            if not allow_in_place:
                image = image.copy()
        if self.flip is not None:
            # Creates view
            image = np.flip(image, tuple(x + image.ndim - self.ndim for x in self.flip))
            if not allow_in_place:
                image = image.copy()
        return image
    
    def call_torch(self, image, allow_in_place=False):
        if self.transpose is not None:
            # Creates view
            image = image.permute(tuple(range(image.ndim - self.ndim)) + tuple(x + image.ndim - self.ndim for x in self.transpose))
            if not allow_in_place:
                image = image.clone()
        if self.flip is not None:
            # Creates copy
            image = torch.flip(image, tuple(x + image.ndim - self.ndim for x in self.flip))

        return image

class ComplexToChannels(UnaryProcessor):
    def __init__(self, channel_dim=1):
        super().__init__()
        self.channel_dim = channel_dim
        
    def call_numpy(self, image, allow_in_place=False):
        return np.concatenate((image.real, image.imag), axis=self.channel_dim)

    def call_torch(self, image, allow_in_place=False):
        return torch.cat((image.real, image.imag), dim=self.channel_dim)
    
class ChannelsToComplex(UnaryProcessor):
    def __init__(self, channel_dim=1):
        super().__init__()
        self.channel_dim = channel_dim
        
    def __call__(self, image, allow_in_place=False):
        index_r = [slice(None)]*(self.channel_dim) + [slice(image.shape[self.channel_dim]//2)]
        index_i = [slice(None)]*(self.channel_dim) + [slice(image.shape[self.channel_dim]//2,None)]
        return image[index_r] + 1j * image[index_i]

class Magnitude(UnaryProcessor):
    def __init__(self):
        super().__init__()
        
    def __call__(self, image, allow_in_place=False):
        return abs(image)

class Threshold(UnaryProcessor):
    def __init__(self, value):
        super().__init__()
        self.value = value
        
    def __call__(self, image, allow_in_place=False):
        return image > self.value

class KeepComplexPhase(UnaryProcessor):
    def __init__(self):
        super().__init__()
        
    def __call__(self, image, allow_in_place=False):
        return image / (abs(image) + 1e-12)

# TODO: Different modes (e.g. real + imag, magn + phase, etc.). Now only does image1 * image2/abs(image2), where image2 is split-complex (first half of channels = real, second half = imag)
class MakeComplex(BinaryProcessor):
    def __init__(self, channel_dim=1):
        super().__init__()
        self.channel_dim = channel_dim
        
    def __call__(self, image1, image2, allow_in_place=False):
        if self.channel_dim == 0:
            nc = image2.shape[0]//2
            phase = (image2[:nc] + 1j * image2[nc:])
            phase /= abs(phase)
            return image1 * phase
        else:
            nc = image2.shape[1]//2
            phase = (image2[:,:nc] + 1j * image2[:,nc:])
            phase /= abs(phase)
            return image1 * phase


class Noise(UnaryProcessor):
    def __init__(self, sigma=0.01):
        super().__init__()
        self.sigma = sigma
        
    def call_numpy(self, image, allow_in_place=False):
        if np.iscomplexobj(image):
            if allow_in_place:
                image += (np.random.randn(*image.shape) + 1j * np.random.randn(*image.shape)) * self.sigma
                return image
            else:
                return image + (np.random.randn(*image.shape) + 1j * np.random.randn(*image.shape)).astype(image.dtype) * self.sigma
        else:
            if allow_in_place:
                image += np.random.randn(*image.shape) * self.sigma
                return image
            else:
                return image + np.random.randn(*image.shape) * self.sigma

    def call_torch(self, image, allow_in_place=False):
        if allow_in_place:
            image += torch.randn(image.shape, dtype=image.dtype, device=image.device) * self.sigma
            return image
        else:
            return image + torch.randn(image.shape, dtype=image.dtype, device=image.device) * self.sigma


class NoiseCov(UnaryProcessor):
    def __init__(self, sigma=0.01):
        super().__init__()
        self.sigma = sigma
        self.torch_cache = None
        
    # TODO: Numpy implementation

    def call_torch(self, image, allow_in_place=False):
        if torch.is_tensor(self.sigma):
            c = torch.linalg.cholesky(self.sigma)
            noise = (torch.randn_like(image.transpose(1,-1)) @ c.T).transpose(-1,1)
        else:
            noise = torch.randn_like(image) * self.sigma
        
        if allow_in_place:
            image += noise
            return image
        else:
            return image + noise


class GenerateGrid2D(NullaryProcessor):
    def __init__(self, batch_size=None, shape=[1,1], dtype='float32', device='cuda:0', torch=False):
        super().__init__(torch=torch)
        self.batch_size = batch_size
        self.shape = shape
        
        self.device = device
        
        if torch:
            self.dtype = convert_dtype_torch(dtype)
        else:
            self.dtype = convert_dtype_numpy(dtype)
            
    def call_numpy(self):
        grid = np.empty(((self.batch_size,) if self.batch_size is not None else ()) + (2,) + tuple(self.shape), dtype=self.dtype)
        grid[:,0] = np.linspace(-1,1,self.shape[1])[None,None,:]
        grid[:,1] = np.linspace(-1,1,self.shape[0])[None,:,None]
        
        return grid
        
    def call_torch(self):
        grid = torch.empty(((self.batch_size,) if self.batch_size is not None else ()) + (2,) + tuple(self.shape), dtype=self.dtype, device=self.device)
        grid[:,0] = torch.linspace(-1,1,self.shape[1])[None,None,:]
        grid[:,1] = torch.linspace(-1,1,self.shape[0])[None,:,None]
        
        return grid


class GenerateNoise(NullaryProcessor):
    def __init__(self, shape=[1], vmin=0, vmax=1, dtype='float32', device='cuda:0', torch=False):
        super().__init__(torch=torch)
        self.vmin = vmin
        self.vmax = vmax
        self.range = vmax-vmin
        self.shape = shape
        self.complex = is_complex_dtype(dtype)
        self.device = device
        
        if torch:
            self.dtype = convert_dtype_torch(dtype)
        else:
            self.dtype = convert_dtype_numpy(dtype)
        
    def call_numpy(self):
        if self.complex:
            return (np.random.rand(*self.shape) + 1j * np.random.rand(*self.shape)).astype(self.dtype) * self.range + self.vmin
        else:
            return np.random.rand(*self.shape).astype(self.dtype) * self.range + self.vmin

    def call_torch(self):
        return torch.rand(tuple(self.shape), dtype=self.dtype, device=self.device) * self.range + self.vmin

class GenerateNormalNoise(NullaryProcessor):
    def __init__(self, shape=[1], mean=0, sigma=1, dtype='float32', device='cuda:0', torch=False):
        super().__init__(torch=torch)
        self.mean = mean
        self.sigma = sigma
        self.shape = shape
        self.complex = is_complex_dtype(dtype)
        self.device = device
        
        if torch:
            self.dtype = convert_dtype_torch(dtype)
        else:
            self.dtype = convert_dtype_numpy(dtype)
        
    def call_numpy(self):
        if self.complex:
            return (np.random.randn(*self.shape) + 1j * np.random.randn(*self.shape)).astype(self.dtype) * self.sigma + self.mean
        else:
            return np.random.randn(*self.shape).astype(self.dtype) * self.sigma + self.mean

    def call_torch(self):
        return torch.randn(tuple(self.shape), dtype=self.dtype, device=self.device) * self.sigma + self.mean



class StackDimensions(UnaryProcessor):
    def __init__(self, dims=[], output_dim=None):
        super().__init__()
        
        self.dims = dims
        if not output_dim:
            output_dim = min(dims)
        self.output_dim = output_dim
    
    def calc_dims(self, ndims):
        not_dims = [x for x in range(ndims) if x not in self.dims]
        self.transpose = not_dims[:self.output_dim] + self.dims + not_dims[self.output_dim:]
        self.consecutive = self.transpose == list(range(len(self.transpose)))
        self.shape_mult = [[x] for x in not_dims[:self.output_dim]] + [self.dims] + [[x] for x in not_dims[self.output_dim:]]
    
    def calc_shape(self, shape):
        output_shape = []
        for x in self.shape_mult:
            s = 1
            for y in x:
                s *= shape[y]
            output_shape.append(s)
        return output_shape
            
    def call_numpy(self, image, allow_in_place=False):
        self.calc_dims(image.ndim)
        output_shape = self.calc_shape(image.shape)
        
        if not self.consecutive:
            image = image.transpose(*self.transpose)

        image = image.reshape(*output_shape)
        # if not allow_in_place:
        #     image = image.copy()
        return image

    def call_torch(self, image, allow_in_place=False):
        self.calc_dims(image.ndim)
        output_shape = self.calc_shape(image.shape)
        
        if not self.consecutive:
            image = image.permute(*self.transpose)

        image = image.reshape(*output_shape)
        # if not allow_in_place:
        #     image = image.copy()
        return image
    


class SplitChannels(UnaryProcessor):
    def __init__(self, shape=[-1], channel_dim=1):
        super().__init__()
        
        self.channel_dim = channel_dim
        self.shape = shape

    def calc_reshape(self, shape):
        return shape[:self.channel_dim] + tuple(self.shape) + shape[self.channel_dim+1:]
        
    # TODO: in-place
    def __call__(self, image, allow_in_place=False):
        return image.reshape(self.calc_reshape(image.shape))

