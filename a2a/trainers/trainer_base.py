# TODO: Allow batches_per_epoch=None to indicate epoch=one pass over dataset
# TODO: Generator needs to support this (if set up as generator, it needs to be recreated over and over (or give an end of dataset signal, e.g. data['end'] == True?))

class Trainer:
    def __init__(self, model, epochs=1, batches_per_epoch=1, validation_batches_per_epoch=1, callbacks=[]):
        self.epochs = epochs
        self.model = model
        self.callbacks = callbacks
        self.batches_per_epoch = batches_per_epoch
        self.validation_batches_per_epoch = validation_batches_per_epoch
    
    def train_batch(self):
        pass
    
    def validate_batch(self):
        pass
    
    def train(self):
        for cb in self.callbacks:
            cb.on_training_start({'model':self.model, 'epochs':self.epochs})
            
        for epoch in range(0, self.epochs):
            epoch_dict = {'model':self.model, 'epoch':epoch, 'epochs':self.epochs}
            
            for cb in self.callbacks:
                cb.on_epoch_start(epoch_dict)

            # Training
            epoch_loss = {}
            total_samples = 0
            
            for iteration in range(self.batches_per_epoch):
                epoch_dict = {'model':self.model, 'epoch':epoch, 'epochs':self.epochs, 'batch':iteration, 'batches':self.batches_per_epoch}
                for cb in self.callbacks:
                    cb.on_batch_start(epoch_dict)
    
                loss, n_samples = self.train_batch()
                loss = {x: loss[x]*n_samples for x in loss}
                total_samples += n_samples
                
                if epoch_loss == {}:
                    epoch_loss = loss
                else:
                    epoch_loss = {x: (epoch_loss[x] if x in epoch_loss else 0) + (loss[x] if x in loss else 0) for x in set(loss) | set(epoch_loss)}
        
            epoch_loss = {x: epoch_loss[x]/total_samples for x in epoch_loss}
            epoch_dict['training_loss'] = epoch_loss
    
            # Validation
            epoch_loss = {}
            total_samples = 0

            for i in range(self.validation_batches_per_epoch):
                # TODO: Would be nice if a None batch would break this loop (e.g. validation generator can indicate end of validation set)
                loss, n_samples = self.validate_batch()
                loss = {l: loss[l]*n_samples for l in loss}
                total_samples += n_samples

                if epoch_loss == {}:
                    epoch_loss = loss
                else:
                    # epoch_loss = {x: epoch_loss[x] + loss[x] for x in loss}
                    epoch_loss = {x: (epoch_loss[x] if x in epoch_loss else 0) + (loss[x] if x in loss else 0) for x in set(loss) | set(epoch_loss)}

            epoch_loss = {x: epoch_loss[x]/total_samples for x in epoch_loss}
            if self.validation_batches_per_epoch > 0:
                epoch_dict['validation_loss'] = epoch_loss
    
            for cb in self.callbacks:
                cb.on_epoch_end(epoch_dict)
                
        for cb in self.callbacks:
            cb.on_training_end({'model':self.model, 'epochs':self.epochs})
