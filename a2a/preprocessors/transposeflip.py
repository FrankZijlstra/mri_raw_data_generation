import numpy as np

from .preprocessor_base import Preprocessor
from . import preprocessor

@preprocessor('TransposeFlip')
class TransposeFlip(Preprocessor):
    def __init__(self, name=None, dataset=[], transpose=None, flip=None, stored_parameters=None):
        super().__init__(name=name, dataset=dataset, stored_parameters=stored_parameters)
        self.transpose = transpose
        self.flip = flip
    
    def process(self, dataset, attrs):
        inverse_parameters = []
        
        for i,d in enumerate(dataset[self.dataset[0]]):
            for x in self.dataset:
                if self.transpose is not None:
                    dataset[x][i] = np.transpose(dataset[x][i], (0,) + tuple(self.transpose))
                if self.flip is not None:
                    dataset[x][i] = np.flip(dataset[x][i], self.flip)

        return inverse_parameters

    def process_inverse(self, dataset, attrs, inverse_parameters):
        for i,d in enumerate(dataset[self.dataset[0]]):
            for x in self.dataset:
                if self.flip is not None:
                    dataset[x][i] = np.flip(dataset[x][i], self.flip)
                if self.transpose is not None:
                    dataset[x][i] = np.transpose(dataset[x][i], np.argsort((0,) + tuple(self.transpose)))
