class Callback:
    def __init__(self, **kwargs):
        pass

    def on_training_start(self, epoch_dict):
        pass

    def on_training_end(self, epoch_dict):
        pass

    def on_epoch_start(self, epoch_dict):
        pass

    def on_epoch_end(self, epoch_dict):
        pass

    def on_batch_start(self, epoch_dict):
        pass
