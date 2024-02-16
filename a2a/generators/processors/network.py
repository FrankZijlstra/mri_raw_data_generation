from a2a.generators.factory import processor
from .processor import Processor

from a2a.models.io import load_from_file

@processor('Model')
class ModelProcessor(Processor):
    models = {}
    def __init__(self, filename, dataset=[], output_dataset=[], datasets=None, device='cuda:0'):
        super().__init__(dataset=dataset, output_dataset=output_dataset)
        if filename not in ModelProcessor.models:
            ModelProcessor.models[filename] = load_from_file(filename)
            ModelProcessor.models[filename].set_requires_grad(ModelProcessor.models[filename].networks.keys(), value=False)

        self.model = ModelProcessor.models[filename]
        self.model.to(device)
        
        self.datasets = datasets
    
    def __call__(self, data, attrs):
        if isinstance(self.dataset, dict):
            data_tmp = {x:data[self.dataset[x]] for x in self.dataset}
        else:
            data_tmp = {x:data[x] for x in self.dataset}
        
        output_data = self.model.apply(data_tmp, datasets=self.datasets)
        
        if isinstance(self.output_dataset, dict):
            for x in self.output_dataset:
                data[x] = output_data[self.output_dataset[x]]
        else:
            for x in self.output_dataset:
                data[x] = output_data[x]

@processor('Rename')
class Rename(Processor):
    def __init__(self, dataset, output_dataset):
        super().__init__(dataset=dataset, output_dataset=output_dataset)
        
        assert(all(x not in output_dataset for x in dataset))
        assert(all(x not in dataset for x in output_dataset))
    
    def __call__(self, data, attrs):
        for x,y in zip(self.dataset, self.output_dataset):
            data[y] = data[x]
            del data[x]
