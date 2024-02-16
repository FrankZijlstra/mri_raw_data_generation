from . import callback
from .callback_base import Callback

from a2a.utils.globals import get_global

def loss_to_str(loss, loss_names, name=True, join_str=' ', precision=4):
    return join_str.join([((n + ': ' if name else '') + '{:.' + str(precision) + 'f}').format(loss[n] if n in loss else 0) for n in loss_names])


@callback('TrainingLogWriter')
class TrainingLogWriter(Callback):
    def __init__ (self, filename='training.log'):
        self.filename = get_global('log_directory') / filename

    def on_training_start(self, epoch_dict):
        self.loss_names = epoch_dict['model'].get_loss_names()
        with open(self.filename, "w") as f:
            f.write('epoch,' + ','.join(self.loss_names) + ',' + ','.join(['val_' + n for n in self.loss_names]) + '\n')

    def on_epoch_end (self, epoch_dict):
        with open(self.filename , "a") as f:
            if 'validation_loss' in epoch_dict:
                f.write('{},{},{}\n'.format(epoch_dict['epoch']+1,loss_to_str(epoch_dict['training_loss'], self.loss_names, join_str=',', name=False, precision=8),loss_to_str(epoch_dict['validation_loss'], self.loss_names, join_str=',', name=False, precision=8)))
            else:
                f.write('{},{},{}\n'.format(epoch_dict['epoch']+1,loss_to_str(epoch_dict['training_loss'], self.loss_names, join_str=',', name=False, precision=8),',' * (len(self.loss_names)-1)))
