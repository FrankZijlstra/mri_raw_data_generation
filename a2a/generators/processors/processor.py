from a2a.generators.factory import processor

from a2a.processors.processor import UnaryProcessor, BinaryProcessor

class Processor:
    def __init__(self, dataset=[], output_dataset=None):
        if isinstance(dataset, str):
            self.dataset = [dataset]
        else:
            self.dataset = dataset
        
        if not output_dataset:
            self.output_dataset = self.dataset
        elif isinstance(output_dataset, str):
            self.output_dataset = [output_dataset]
        else:
            self.output_dataset = output_dataset
        
    def __call__(self, data, attrs):
        pass


def register_op_processor(name, op):
    class UnaryOpProcessor(Processor):
        def __init__(self, dataset=[], output_dataset=None, **kwargs):
            super().__init__(dataset=dataset, output_dataset=output_dataset)

            self.op = op(**kwargs)
        
        def __call__(self, data, attrs):
            for i,o in zip(self.dataset, self.output_dataset):
                data[o] = self.op(data[i]) # allow_in_place if this is not the first processor and i==o
                
    class BinaryOpProcessor(Processor):
        def __init__(self, dataset=[], output_dataset=None, **kwargs):
            super().__init__(dataset=dataset, output_dataset=output_dataset)
            assert(len(self.dataset) == 2)
            assert(len(self.output_dataset) >= 1) # TODO: Multiply dataset: [a,b] fills output_dataset with [a,b]
            self.op = op(**kwargs)
        
        def __call__(self, data, attrs):
            data[self.output_dataset[0]] = self.op(data[self.dataset[0]], data[self.dataset[1]]) # allow_in_place if this is not the first processor and i==o
    
    if issubclass(op, UnaryProcessor):
        processor(name)(UnaryOpProcessor)
    elif issubclass(op, BinaryProcessor):
        processor(name)(BinaryOpProcessor)
    else:
        raise RuntimeError('Wrong op type: ', op)

