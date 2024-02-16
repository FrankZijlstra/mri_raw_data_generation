from . import preprocessor

from a2a.processors.processor import UnaryProcessor, BinaryProcessor, NullaryProcessor


class Preprocessor:
    def __init__(self, name=None, dataset=[], output_dataset=None, stored_parameters=None):
        if isinstance(dataset, str):
            dataset = [dataset]
        if output_dataset is None:
            output_dataset = dataset
        if isinstance(output_dataset, str):
            output_dataset = [output_dataset]
            
        self.dataset = dataset
        self.output_dataset = output_dataset
        self.parameters = stored_parameters
        self.name = name

    def is_deterministic(self):
        return True
    
    def requires_fit(self):
        return False

    # Check if preprocessor should be run on this dataset
    # TODO: Is this check still necessary? (meant for inference? in but not out, but we'd rather have a hand-defined inference pipeline?)
    def check_dataset(self, dataset):
        return all([x in dataset for x in self.dataset])
    
    def fit_process(self, dataset, attrs):
        self.parameters = self.fit(dataset, attrs)
        ip = self.process(dataset, attrs)
        
        return ip, self.parameters

    # Fits parameters to the dataset
    # dataset is a list of dict of volumes of shape C+spatial
    # Return value are parameters to store for applying to test data, return None if not needed
    def fit(self, dataset, attrs):
        return None

    # dataset is a dict of volumes of shape C+spatial
    # Operations on dataset must be in-place
    def process(self, dataset, attrs):
        return None
    
    def process_inverse(self, dataset, attrs, inverse_parameters):
        raise Exception('Inverse processor not available')


def register_op_preprocessor(name, op, **defaults):
    class UnaryOpPreprocessor(Preprocessor):
        def __init__(self, name=None, dataset=[], output_dataset=None, stored_parameters=None, **kwargs):
            super().__init__(name=name, dataset=dataset, output_dataset=output_dataset, stored_parameters=stored_parameters)
            pars = dict(defaults)
            pars.update(kwargs)
            self.op = op(**pars)
        
        def process(self, dataset, attrs):
            for o,x in zip(self.output_dataset, self.dataset):
                dataset[o] = self.op(dataset[x])
                attrs[o] = attrs[x]
            
            return None
        
    class BinaryOpPreprocessor(Preprocessor):
        def __init__(self, name=None, dataset=[], output_dataset=None, stored_parameters=None, **kwargs):
            super().__init__(name=name, dataset=dataset, output_dataset=output_dataset, stored_parameters=stored_parameters)
            pars = dict(defaults)
            pars.update(kwargs)
            assert(len(self.dataset) == 2)
            assert(len(self.output_dataset) >= 1) # TODO: Multiply dataset: [a,b] fills output_dataset with [a,b]
            self.op = op(**pars)
        
        def process(self, dataset, attrs):
            dataset[self.output_dataset[0]] = self.op(dataset[self.dataset[0]], dataset[self.dataset[1]]) # allow_in_place?
            attrs[self.output_dataset[0]] = {}
            return None
    
    class NullaryOpPreprocessor(Preprocessor):
        def __init__(self, name=None, output_dataset=None, stored_parameters=None, **kwargs):
            super().__init__(name=name, dataset=[], output_dataset=output_dataset, stored_parameters=stored_parameters)
            pars = dict(defaults)
            pars.update(kwargs)
            self.op = op(**pars)
        
        def process(self, dataset, attrs):
            for o in self.output_dataset:
                dataset[o] = self.op()
        
            return None
    
    if issubclass(op, UnaryProcessor):
        preprocessor(name)(UnaryOpPreprocessor)
    elif issubclass(op, BinaryProcessor):
        preprocessor(name)(BinaryOpPreprocessor)
    elif issubclass(op, NullaryProcessor):
        preprocessor(name)(NullaryOpPreprocessor)
    else:
        raise RuntimeError('Wrong op type: ', op)
