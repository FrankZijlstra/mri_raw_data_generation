from . import callback
from .callback_base import Callback

import time
import datetime

from .log_writer import loss_to_str


@callback('ProgressPrinter')
class ProgressPrinter(Callback):
    def on_training_start(self, epoch_dict):
        self.training_start_time = time.perf_counter()
        
    def on_training_end(self, epoch_dict):
        dt = datetime.timedelta(seconds=time.perf_counter() - self.training_start_time)
        print(f'Total training time: {dt}')

    def on_epoch_start (self, epoch_dict):
        self.start_time = time.perf_counter()
        print('Epoch: {}'.format(epoch_dict['epoch']+1))

    def on_epoch_end (self, epoch_dict):
        print("Epoch {}: Loss: {}".format(epoch_dict['epoch']+1, loss_to_str(epoch_dict['training_loss'], epoch_dict['model'].get_loss_names())))
        if 'validation_loss' in epoch_dict:
            print("Epoch {}: Val. Loss: {}".format(epoch_dict['epoch']+1, loss_to_str(epoch_dict['validation_loss'], epoch_dict['model'].get_loss_names())))
        print('Time elapsed: {:.2f} seconds'.format(time.perf_counter() - self.start_time))

