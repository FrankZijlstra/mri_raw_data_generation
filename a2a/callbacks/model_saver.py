from . import callback
from .callback_base import Callback

import glob
import os

from a2a.utils.globals import get_global

@callback('ModelSaver')
class ModelSaver(Callback):
    def __init__ (self, interval=1, verbose=False, keep_only_last=False, save_optimizer=False):
        self.log_directory = get_global('log_directory') / 'weights'
        self.interval = interval
        self.verbose = verbose
        self.keep_only_last = keep_only_last
        self.prev_epoch = None
        self.save_optimizer = save_optimizer

    def on_epoch_end (self, epoch_dict):
        if (epoch_dict['epoch']+1) % self.interval == 0:
            if self.keep_only_last and self.prev_epoch is not None:
                for f in glob.glob("{}/epoch_{}*".format(self.log_directory, self.prev_epoch+1)):
                    os.remove(f)

            filename = "{}/epoch_{}".format(self.log_directory, epoch_dict['epoch']+1)
            if self.verbose:
                print('Saving model in epoch {}: {}'.format(epoch_dict['epoch']+1, filename))
            epoch_dict['model'].save(filename, save_optimizer=self.save_optimizer)

            self.prev_epoch = epoch_dict['epoch']

    def on_training_end(self, epoch_dict):
        if self.keep_only_last and self.prev_epoch is not None:
            for f in glob.glob("{}/epoch_{}*".format(self.log_directory, self.prev_epoch+1)):
                    os.remove(f)
