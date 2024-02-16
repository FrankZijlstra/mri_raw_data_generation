import torch
import numpy as np

from . import model_runner

from a2a.processors.gpu import GPU
from a2a.utils.utils import get_random_state, set_random_state, set_seed

@model_runner('Default')
class ModelRunner:
    def __init__(self, model):
        self.model = model

    # image as CxZxYxX
    def __call__(self, dataset):
        with torch.no_grad():
            output = []
            op = GPU(device=self.model.get_device())
            for image in dataset[self.dataset]:
                output.append(self.model(op(image[np.newaxis,...])).cpu().numpy())
            return output


@model_runner('DefaultSeeded', processor_parameters=['processors'])
class DefaultSeeded:
    def __init__(self, model, processors=[]):
        self.model = model
        self.processors = processors

    # image as CxZxYxX
    def __call__(self, dataset, attr, seed=0):
        data = dict(dataset) # C + spatial?
        attr = dict(attr)
        
        s = get_random_state()
        set_seed(seed)
        for processor in self.processors:
            processor(data, attr)
        
        # TODO: Each dataset should add a batch dim of 1?
        out = self.model.apply(data)
        set_random_state(s)
        # TODO: What to do with larger batches? -> Model runner should processor C + spatial, no batch dim
        # TODO: But a generative model we'd like to run with larger batch_size

        return {x:out[x] for x in out} 
