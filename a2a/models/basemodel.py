import torch
import torch.nn as nn

from .model_base import Model
from . import model
                        
@model('Base', network_parameters=['network'], optimizer_parameters={'optimizer':['network']}, loss_parameters=['loss'])
class BaseModel(Model):
    def __init__(self, network, optimizer, input_dataset=None, output_dataset=None, loss=nn.L1Loss(), clip_gradient_norm=0, parameters={}):
        super().__init__(networks={'net':network}, optimizers={'opt':optimizer}, parameters=parameters)
        
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        
        self.training_passes = [{'optimizer':'opt',
                                 'networks':['net'],
                                 'compute':self.training_pass,
                                 'loss_name':'loss'}]
        
        self.loss = loss
        self.clip_gradient_norm = clip_gradient_norm
        
    def clean(self):
        del self.output
        self.losses = {}
        
    def forward(self, input):
        self.output = self.networks['net'](input)
    
    def apply(self, data, datasets=None):
        if datasets is None:
            datasets = [self.output_dataset]
        
        d = {}
        self.set_eval_mode()
        if self.output_dataset in datasets:
            input = data[self.input_dataset]
            with torch.no_grad():
                if isinstance(datasets, dict):
                    d[datasets[self.output_dataset]] = self.networks['net'](input)
                else:
                    d[self.output_dataset] = self.networks['net'](input)
        return d
        
    def training_pass(self, data):
        input = data[self.input_dataset]
        output = data[self.output_dataset]
        self.forward(input)
        self.record_loss('loss', '', 1, self.loss(self.output, output))
        
    def get_loss_names(self):
        return ['loss']
    
