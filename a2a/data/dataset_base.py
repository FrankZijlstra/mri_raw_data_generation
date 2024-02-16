from . import dataset
from a2a.utils.globals import get_global

class ReadOnlyDict(dict):
    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("Cannot modify ReadOnlyDict")
    __setitem__ = __readonly__
    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__

def run_preprocessors(d,a,preprocessors):
    # TODO: Mind overwriting dict items, dict might come directly from dataset
    
    inverse_parameters = []

    for p in preprocessors:
        ip = p.process(d, a)
        inverse_parameters.append(ip)

    return inverse_parameters

def name_preprocessors(preprocessors, prefix=None):
    counters = {}
    names = []
    for x in preprocessors:
        if x.name:
            name = x.name
        else:
            n = type(x).__name__
        
            if n in counters:
                counters[n] += 1
            else:
                counters[n] = 1
            
            name = f'{n}_{counters[n]}'
        names.append(name)

    if prefix:
        names = [prefix + '.' + x for x in names]
    return names   

@dataset('Dataset', preprocessor_parameters=['preprocessors'])
class Dataset:
    def __init__(self, fold=0, preprocessors=[], preprocessors_parameters=None, preprocessor_name_base=None):
        self.fold = fold
        self.preprocessor_name_base = preprocessor_name_base
        
        self.preprocessors = preprocessors
        self.name_preprocessors()
        
        self.preprocessors_require_fit = preprocessors_parameters is None and any(x.requires_fit() for x in self.preprocessors)
        self.preprocessors_parameters = preprocessors_parameters
        self.split_preprocessors()
        
        if preprocessors_parameters:
            self.fill_preprocessors_parameters(preprocessors_parameters)
    
    def fill_preprocessors_parameters(self, preprocessors_parameters):
        if preprocessors_parameters in get_global('datasets') and hasattr(get_global('datasets')[preprocessors_parameters], 'stored_parameters'):
            p = get_global('datasets')[preprocessors_parameters].stored_parameters
        else:
            p = {}

        for x in self.preprocessors:
            if x.requires_fit():
                if x.name in p:
                    x.parameters = p[x.name]
                else:
                    raise RuntimeError(f'Fitted parameters not found for {x.name}')
    
    def name_preprocessors(self):
        counters = {}
        for x in self.preprocessors:
            if not x.name:
                n = type(x).__name__
            
                if n in counters:
                    counters[n] += 1
                else:
                    counters[n] = 1
                
                x.name = f'{n}_{counters[n]}'
                if self.preprocessor_name_base:
                    x.name = self.preprocessor_name_base + '.' + x.name
    
    def split_preprocessors(self):
        self.preprocessors_deterministic = []
        i = 0
        while i < len(self.preprocessors) and self.preprocessors[i].is_deterministic():
            self.preprocessors_deterministic.append(self.preprocessors[i])
            i += 1
        
        self.preprocessors_nondeterministic = []
        while i < len(self.preprocessors):
            self.preprocessors_nondeterministic.append(self.preprocessors[i])
            i += 1
    
    def fit_preprocessors(self, data, attr):
        if not self.preprocessors_require_fit:
            return
        
        inverse_parameters = {}
        stored_parameters = {}
        
        data = [dict(x) for x in data]
        attr = [dict(x) for x in attr]
        
        # Find out which is the last preprocessor that requires a fit, only run pipeline until that point
        last_fit_index = max(zip((x.requires_fit() for x in self.preprocessors), range(len(self.preprocessors))))[1] + 1

        for p in self.preprocessors[:last_fit_index]:
            print('Fitting', p.name)
            sp = p.fit(data, attr)
            print([(x, data[0][x].dtype) for x in data[0]])
            ips = []
            for d,a in zip(data, attr):
                ip = p.process(d, a)
                ips.append(ip)
            
            if any(ips):
                inverse_parameters[p.name] = ips
            if sp:
                stored_parameters[p.name] = sp
        
        self.stored_parameters = stored_parameters

    def preprocess_item(self, d, a):
        run_preprocessors(d, a, self.preprocessors_deterministic)
        run_preprocessors(d, a, self.preprocessors_nondeterministic)
        
    def load(self):
        # TODO: If preprocessors are generative, can require fit
        pass
    
    def is_deterministic(self):
        return len(self.preprocessors_nondeterministic) == 0
    
    def generator(self, infinite=True, random=True, shuffle=False):
        if random:
            while infinite:
                yield self.random_item()
        else:
            while infinite:
                for x in self.iterator(shuffle=shuffle):
                    yield x

    def iterator(self, shuffle=False):
        raise NotImplementedError('Dataset does not support iterator')
    
    def indexed_item(self, index):
        raise NotImplementedError('Dataset does not support item indexing')

    def number_of_items(self):
        raise NotImplementedError('Dataset does not support number of items')
    
    def random_item(self):
        d = {}
        a = {}
        self.preprocess_item(d, a)
        return d, a
