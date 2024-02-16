import torch
import gc

from . import trainer
from .trainer_base import Trainer

from a2a.utils.globals import get_global

@trainer('Paired', callback_parameters=['callbacks'])
class PairedTrainer(Trainer):
    def __init__(self, model, epochs=1, train_generator_name='train', validation_generator_name=None, batches_per_epoch=1, validation_batches_per_epoch=1,  callbacks=[]):
        super().__init__(model, epochs=epochs, batches_per_epoch=batches_per_epoch, validation_batches_per_epoch=validation_batches_per_epoch, callbacks=callbacks)
        generators = get_global('generators')
        self.train_generator = generators[train_generator_name]
        if validation_generator_name:
            self.validation_generator = generators[validation_generator_name]
        else:
            self.validation_batches_per_epoch = 0           

    def train_batch(self):
        d = next(self.train_generator)

        self.model.train(d)
        loss = self.model.get_losses()
        n_samples = list(d.values())[0].shape[0] # Assume all data entries have the same batch size (TODO: model could keep track of last batch size)
        
        self.model.clean()
        # del d
        # gc.collect()
        
        return loss, n_samples
    
    def validate_batch(self):
        d = next(self.validation_generator)
        
        self.model.validate(d)
        loss = self.model.get_losses()
        n_samples = list(d.values())[0].shape[0] # Assume all data entries have the same batch size
        
        self.model.clean()
        # del d
        # gc.collect()
        
        return loss, n_samples
