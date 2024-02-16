import random

from . import dataset
from . import dataloader

from .dataset_base import Dataset, run_preprocessors, ReadOnlyDict


# Static dataset loads all available data into memory
# TODO: Needs mode to generate non-deterministic data only once (i.e. for a static validation set)
@dataset('StaticDataset', preprocessor_parameters=['preprocessors'])
class StaticDataset(Dataset):
    def __init__(self, fold=0, dataset={}, preprocessors=[], preprocessors_parameters=None, preprocessor_name_base=None, indices=None, nondeterministic_loading=False):
        super().__init__(fold=fold, preprocessors=preprocessors, preprocessors_parameters=preprocessors_parameters, preprocessor_name_base=preprocessor_name_base)
        
        self.data = {}
        self.attr = {}
        
        self.nondeterministic_loading = nondeterministic_loading
        self.loaders = {x:dataloader.create(dataset[x]) for x in dataset} # TODO: dataset could be dataloader_parameters if apply_on_all is fixed to also accept dict (without type)
        
        if indices:
            if type(indices) is not list:
                raise ValueError('TODO: Implement CV/number of items option for indices')    
            else:
                self.indices = indices
        else:
            # TODO: valid_indices can be None (i.e. Dynamic/random loader)
            inds = [self.loaders[x].get_valid_indices() for x in self.loaders]
            if len(inds) > 0:
                self.indices = [[x for x in inds[0] if all(x in y for y in inds)]]
            else:
                self.indices = []
        
        self.folds = len(self.indices)
        self.validate_indices()
        
        if self.fold > self.folds:
            raise ValueError('Invalid CV fold')
        
    
    def validate_indices(self):
        for x in self.loaders:
            valid_inds = self.loaders[x].get_valid_indices()
            
            for inds in self.indices:
                for i in inds:
                    # TODO: Explore conversions, i.e. '1' != 1
                    if i not in valid_inds:
                        raise RuntimeError(f'Invalid index in {__class__}: {i} not in dataset \'{x}\'')

    def number_of_items(self):
        return len(self.data)
    
    def load(self):       
        data = {}
        attr = {}
        
        for dataset_name in self.loaders:
            print('Loading', dataset_name)
            data[dataset_name], attr[dataset_name] = self.loaders[dataset_name].load_dataset(self.indices[self.fold])
      
        self.data = []
        self.attr = []
        
        if data != {}:
            for i in range(len(data[list(data.keys())[0]])):
                self.data.append({x:data[x][i] for x in data})
                self.attr.append({x:attr[x][i] for x in attr})
        else:
            self.data.append({})
            self.attr.append({})
        
        self.fit_preprocessors(self.data, self.attr)
        
        for d,a in zip(self.data, self.attr):
            run_preprocessors(d, a, self.preprocessors_deterministic)
            if self.nondeterministic_loading:
                run_preprocessors(d, a, self.preprocessors_nondeterministic)
        
        self.data = [ReadOnlyDict(x) for x in self.data]
        self.attr = [ReadOnlyDict(x) for x in self.attr]

    def preprocess_item(self, d, a):
        if not self.nondeterministic_loading and self.preprocessors_nondeterministic != []:
            run_preprocessors(d, a, self.preprocessors_nondeterministic)
    
    def indexed_item(self, index):
        return dict(self.data[index]), dict(self.attr[index])
    
    def random_item(self):
        i = random.randint(0,len(self.data)-1)
        d = dict(self.data[i])
        a = dict(self.attr[i])
        self.preprocess_item(d, a)
        return d, a
    
    def iterator(self, shuffle=False):
        indices = list(range(self.number_of_items()))
        if shuffle:
            random.shuffle(indices)
        
        for i in indices:
            d,a = self.indexed_item(i)
            self.preprocess_item(d, a)
            yield d, a