import random
import time
import threading
import weakref

from . import dataset

from .dataset_base import run_preprocessors, ReadOnlyDict
from .static_dataset import StaticDataset


@dataset('DynamicDataset', preprocessor_parameters=['preprocessors'])
class DynamicDataset(StaticDataset):
    def __init__(self, fold=0, dataset={}, preprocessors=[], preprocessors_parameters=None, preprocessor_name_base=None, indices=None, n_preload=0, n_data=0, n_reload_workers=0, reload_delay=10, reload_interval=1000, shuffle=False, nondeterministic_loading=False):
        super().__init__(fold=fold, dataset=dataset, preprocessors=preprocessors, preprocessors_parameters=preprocessors_parameters, preprocessor_name_base=preprocessor_name_base, indices=indices, nondeterministic_loading=nondeterministic_loading)
        self.n_preload = n_preload
        self.n_data = n_data
        
        self.n_reload_workers = n_reload_workers
        self.reload_delay = reload_delay
                
        self.reload_interval = reload_interval
        self.reload_counter = 0
        
        self.index_gen = self.index_generator(shuffle=shuffle)

    def number_of_items(self):
        if not self.indices[self.fold]:
            raise RuntimeError('Dynamic dataset does not support number of items') # TODO: Could just return 0 or None
        
        return len(self.indices[self.fold])

    def index_generator(self, shuffle=False):
        while True:
            if not self.indices[self.fold]:
                yield None
            
            indices = list(range(self.number_of_items()))
            if shuffle:
                random.shuffle(indices)
                
            yield from indices

    def load_and_replace(self):
        (d,a),idx = self.load_next()
        
        run_preprocessors(d, a, self.preprocessors_deterministic)
        
        if self.nondeterministic_loading:
            run_preprocessors(d, a, self.preprocessors_nondeterministic)
        
        replace_index = random.randint(0,len(self.data)-1)

        self.data[replace_index] = ReadOnlyDict(d)
        self.attr[replace_index] = ReadOnlyDict(a)
        self.data_indices[replace_index] = idx

    def run(self):
        try:
            while True:
                time.sleep(self.reload_delay)
                self.load_and_replace()
            
        except ReferenceError:
            pass
    
    def load_indexed(self, index):
        data = {}
        attr = {}

        for x in self.loaders:
            d,a = self.loaders[x].load(self.indices[self.fold][index])

            data[x] = d
            attr[x] = a

        return data, attr
    
    def indexed_item(self, index):
        if index in self.data_indices:
            i = self.data_indices.index(index)
            return dict(self.data[i]), dict(self.attr[i])
        else:
            d,a = self.load_indexed(index)

            run_preprocessors(d, a, self.preprocessors_deterministic)
            if self.nondeterministic_loading:
                run_preprocessors(d, a, self.preprocessors_nondeterministic)

            if self.n_data != 0 and len(self.data) == self.n_data:
                replace_index = random.randint(0,len(self.data)-1)

                self.data[replace_index] = ReadOnlyDict(d)
                self.attr[replace_index] = ReadOnlyDict(a)
                self.data_indices[replace_index] = index
                
            return d, a
    
    def load_next(self):
        idx = next(self.index_gen)
        return self.load_indexed(idx), idx

    def load(self):        
        self.data = []
        self.attr = []
        self.data_indices = []
        
        for i in range(self.n_preload):
            idx = next(self.index_gen)
            d, a = self.load_indexed(idx)
            self.data.append(d)
            self.attr.append(a)
            self.data_indices.append(idx)
        
        # TODO: Fit only when training, i.e. this could just load pre-fitted parameters
        self.fit_preprocessors(self.data, self.attr)
        
        if self.n_data > self.n_preload:
            for i in range(self.n_data - self.n_preload):
                (d, a),idx = self.load_next()
                self.data.append(d)
                self.attr.append(a)
                self.data_indices.append(idx)
        else:
            self.data = self.data[:self.n_data]
            self.attr = self.attr[:self.n_data]
            self.data_indices = self.data_indices[:self.n_data]
        
        for d,a in zip(self.data, self.attr):
            run_preprocessors(d, a, self.preprocessors_deterministic)
            if self.nondeterministic_loading:
                run_preprocessors(d, a, self.preprocessors_nondeterministic)

    
        self.data = [ReadOnlyDict(x) for x in self.data]
        self.attr = [ReadOnlyDict(x) for x in self.attr]
        
        if self.n_reload_workers > 0:
            self.worker_threads = [threading.Thread(target=DynamicDataset.run, args=(weakref.proxy(self),), daemon=True) for x in range(self.n_reload_workers)]
            for x in self.worker_threads:
                x.start()

    def random_item(self):
        if self.n_reload_workers == 0:
            self.reload_counter += 1
            if self.reload_counter == self.reload_interval:
                self.reload_counter = 0
                self.load_and_replace()
        
        if len(self.data) > 0:
            i = random.randint(0,len(self.data)-1)
            d = dict(self.data[i])
            a = dict(self.attr[i])
        else:
            (d,a),idx = self.load_next()
        self.preprocess_item(d, a)
        return d, a

    def iterator(self, shuffle=False):
        # TODO: Indexed item in generator now always loads the data on demand

        if not self.indices[self.fold]:
            raise RuntimeError('Dynamic dataset cannot be iterated over')
        
        return super().iterator(shuffle=shuffle)
