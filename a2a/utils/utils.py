import numpy as np
import torch
import random



def get_random_state():
    return {'random': random.getstate(), 'numpy':np.random.get_state(), 'torch':torch.random.get_rng_state(), 'torch_cuda':torch.cuda.get_rng_state_all()}
    
def set_random_state(state):
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def is_numpy(x):
    return not torch.is_tensor(x)
