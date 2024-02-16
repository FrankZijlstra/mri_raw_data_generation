from . import callback
from .callback_base import Callback


@callback('ParameterScheduler')
class ParameterScheduler(Callback):
    def __init__(self, schedule, verbose=False):
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_start(self, epoch_dict):
        e = epoch_dict['epoch']
        if e in self.schedule:
            if self.verbose:
                print('Setting parameters: '+str(self.schedule[e]))

            epoch_dict['model'].set_parameters(self.schedule[e])
