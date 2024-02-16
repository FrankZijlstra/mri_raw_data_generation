from . import callback
from .callback_base import Callback

import numpy as np

@callback('DictionaryLRScheduler')
class DictionaryLRScheduler(Callback):
    def __init__(self, schedule, verbose=False):
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_start(self, epoch_dict):
        e = epoch_dict['epoch']
        if e in self.schedule:
            if self.verbose:
                print('Setting LR: '+str(self.schedule[e]))

            epoch_dict['model'].set_learning_rate(self.schedule[e])


@callback('LinearLRScheduler')
class LinearLRScheduler(DictionaryLRScheduler):
    def __init__(self, learning_rate=0.01, values={0:1, 1:0}, percentage=True, absolute=False, verbose=False):
        self.learning_rate = learning_rate
        self.values = values
        self.percentage = percentage
        self.absolute = absolute

        super().__init__({}, verbose=verbose)

    def on_training_start(self, epoch_dict):
        epochs = epoch_dict['epochs']
        
        values = {}
                
        for x in self.values:
            if self.percentage:
                e = x * (epochs - 1)
            else:
                e = x
            v = self.values[x]
            if isinstance(v, dict):
                values[e] = {}
                for y in v:
                    values[e][y] = v[y]
            else:
                values[e] = v
        
        if any(isinstance(values[x],dict) for x in values):
            keys = set()
            if isinstance(self.learning_rate, dict):
                keys |= self.learning_rate.keys()
            for x in values:
                if isinstance(values[x],dict):
                    keys |= values[x].keys()
            
            values_new = {}
            for k in keys:
                values_new[k] = {}
                for x in values:
                    if isinstance(values[x],dict):
                        if k in values[x]:
                            values_new[k][x] = values[x][k]
                    else:
                        values_new[k][x] = values[x]

            if not self.absolute:
                for k in values_new:
                    if isinstance(self.learning_rate, dict):
                        lr = self.learning_rate[k]
                    else:
                        lr = self.learning_rate
                        
                    values_new[k] = {x:lr*values_new[k][x] for x in values_new[k]}
    
            schedule = {}
            for k in values_new:
                xs = sorted(values_new[k].keys())
                ys = [values_new[k][x] for x in xs]
                schedule[k] = dict(zip(range(epochs), np.interp(range(epochs), xs, ys)))
            
            self.schedule = {}
            for x in range(epochs):
                self.schedule[x] = {k:schedule[k][x] for k in keys}
            
        else:
            xs = sorted(values.keys())
            ys = [values[x] for x in xs]
            
            schedule = dict(zip(range(epochs), np.interp(range(epochs), xs, ys)))
            
            if not self.absolute and isinstance(self.learning_rate, dict):
                self.schedule = {}
                for x in schedule:
                    self.schedule[x] = {k:schedule[x] * self.learning_rate[k] for k in self.learning_rate}
            else:
                self.schedule = {x:schedule[x] * self.learning_rate for x in schedule}
