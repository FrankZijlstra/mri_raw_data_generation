import torch
import numpy as np

# Operations:
    # Generator processor operates on BC + spatial (but it adds overhead to process every B separately for most operations)
    # Preprocessor operates on C + spatial

# Operations should be able to operate on CPU and GPU where possible
# Assumption: Every operation knows its dimensionality, work on last n dims

def is_numpy(x):
    return not torch.is_tensor(x)

# TODO: Additional dtypes (ints, bfloat?)

def convert_dtype_torch(x):
    return {'float16':torch.float16,
            'float32':torch.float32,
            'float64':torch.float64,
            'complex64':torch.complex64,
            'complex128':torch.complex128}[x]

def convert_dtype_from_torch(x):
    return {torch.float16:'float16',
            torch.float32:'float32',
            torch.float64:'float64',
            torch.complex64:'complex64',
            torch.complex128:'complex128'}[x]

def convert_dtype_numpy(x):
    return {'float16':np.float16,
            'float32':np.float32,
            'float64':np.float64,
            'complex64':np.complex64,
            'complex128':np.complex128}[x]

def convert_dtype_from_numpy(x):
    return str(x)

def is_complex_dtype(x):
    return {'float16':False,
            'float32':False,
            'float64':False,
            'complex64':True,
            'complex128':True}[x]

def is_complex(x):
    if is_numpy(x):
        return is_complex_dtype(convert_dtype_from_numpy(x.dtype))
    else:
        return is_complex_dtype(convert_dtype_from_torch(x.dtype))


class NullaryProcessor:
    def __init__(self, torch=False):
        self.torch = torch

    def __call__(self):        
        if not self.torch:
            return self.call_numpy()
        else:
            return self.call_torch()
        
    def call_numpy(self):
        raise NotImplementedError(f'Numpy operation for {self.__class__} is not implemented')
    def call_torch(self):
        raise NotImplementedError(f'Torch operation for {self.__class__} is not implemented')

class UnaryProcessor:
    def __call__(self, image, allow_in_place=False):        
        if is_numpy(image):
            return self.call_numpy(image, allow_in_place=allow_in_place)
        else:
            return self.call_torch(image, allow_in_place=allow_in_place)

    def call_numpy(self, image, allow_in_place=False):
        raise NotImplementedError(f'Numpy operation for {self.__class__} is not implemented')
    def call_torch(self, image, allow_in_place=False):
        raise NotImplementedError(f'Torch operation for {self.__class__} is not implemented')
        
# If types of inputs do not match, converts to torch if prefer_torch else to numpy
class BinaryProcessor:
    def __init__(self, prefer_torch=True):
        self.prefer_torch = prefer_torch
        
    def call_numpy(self, image1, image2, allow_in_place=False):
        raise NotImplementedError(f'Numpy operation for {self.__class__} is not implemented')
        
    def call_torch(self, image1, image2, allow_in_place=False):
        raise NotImplementedError(f'Torch operation for {self.__class__} is not implemented')
        
    def __call__(self, image1, image2, allow_in_place=False):
        np1 = is_numpy(image1)
        np2 = is_numpy(image2)
        
        if np1 and np2:
            return self.call_numpy(image1, image2, allow_in_place=allow_in_place)
        elif not np1 and not np2:
            return self.call_torch(image1, image2, allow_in_place=allow_in_place)
        else:
            if self.prefer_torch:
                if np1:
                    return self.call_torch(torch.from_numpy(image1).to(device=image2.device, non_blocking=True), image2, allow_in_place=allow_in_place)
                else:
                    return self.call_torch(image1, torch.from_numpy(image2).to(device=image1.device, non_blocking=True), allow_in_place=allow_in_place)
            else:
                if np1:
                    return self.call_numpy(image1, image2.detach().cpu().numpy(), allow_in_place=allow_in_place)
                else:
                    return self.call_numpy(image1.detach().cpu().numpy(), image2, allow_in_place=allow_in_place)

