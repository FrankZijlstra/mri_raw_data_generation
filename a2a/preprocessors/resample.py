from .preprocessor_base import Preprocessor
from . import preprocessor

from a2a.processors.transform import Resample as ResampleOp
@preprocessor('Resample')
class Resample(Preprocessor):
    def __init__(self, name=None, dataset=[], output_dataset=None, target_shape=[], stored_parameters=None):
        super().__init__(name=name, dataset=dataset, output_dataset=output_dataset, stored_parameters=stored_parameters)
        self.target_shape = target_shape
        if type(target_shape) is float:
            target_shape = [1,1]
        self.op = ResampleOp(target_shape=target_shape)
        
    
    def process(self, dataset, attrs):
        if type(self.target_shape) is float:
            self.op.target_shape = tuple(round(y*self.target_shape) for y in dataset[self.dataset[0]].shape[2:])
            
        inverse_parameters = {}
        for o,x in zip(self.output_dataset, self.dataset):
            inverse_parameters[o] = dataset[x].shape
            dataset[o] = self.op(dataset[x])

        return inverse_parameters

    def process_inverse(self, dataset, attrs, inverse_parameters):
        # TODO: Inverse should consider output_dataset. Better yet, refactor inverse processing as a separate pipeline with shared parameters
        for i,d in enumerate(dataset[self.dataset[0]]):
            for x in self.dataset:
                dataset[x][i] = ResampleOp(target_shape=inverse_parameters[i][x])(dataset[x][i])

from a2a.processors.transform import ResampleAvg as ResampleAvgOp
@preprocessor('ResampleAvg')
class ResampleAvg(Preprocessor):
    def __init__(self, name=None, dataset=[], output_dataset=None, target_shape=[], stored_parameters=None):
        super().__init__(name=name, dataset=dataset, output_dataset=output_dataset, stored_parameters=stored_parameters)
        self.op = ResampleAvgOp(target_shape=target_shape)
        self.target_shape = target_shape
    
    def process(self, dataset, attrs):
        inverse_parameters = {}
        for o,x in zip(self.output_dataset, self.dataset):
            inverse_parameters[o] = dataset[x].shape
            dataset[o] = self.op(dataset[x])

        return inverse_parameters
