import torch
import torch.nn as nn
from torch import autograd
import yaml

from .model_base import Model, optimizer_to
from . import model
from .replaybuffer import ReplayBuffer

from .model_base import Discriminator

@model('AAE', network_parameters=['networks'], optimizer_parameters={'optimizers':{'G':[{'networks': ['E', 'G', 'Z']}], 'D':[{'networks':['D']}]}})
class AAE(Model):
    def __init__(self, networks={}, optimizers={}, mask_dataset=None, input_dataset=None, output_dataset=None, latent_dataset=None, clip_gradient_norm=0, d_steps=1, parameters={}, wgan=False):
        self.parameters = {'lambda_D_D':1, 'lambda_G_D': 1, 'lambda_GP':0, 'lambda_rec': 1, 'lambda_Z_rec': 1}
        super().__init__(networks=networks, optimizers=optimizers, parameters=parameters)
        
        if 'E' not in networks:
            raise ValueError('Network E must be supplied')
        if 'Z' not in networks:
            raise ValueError('Network Z must be supplied')
        if 'G' not in networks:
            raise ValueError('Network G must be supplied')
        if 'D' not in networks:
            raise ValueError('Network D must be supplied')
        
        if 'D' not in optimizers:
            raise ValueError('Optimizer D must be supplied')
        if 'G' not in optimizers:
            raise ValueError('Optimizer G must be supplied')
       
        self.clip_gradient_norm = clip_gradient_norm
        
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        self.latent_dataset = latent_dataset
        self.mask = mask_dataset
        
        self.d_steps = d_steps
        self.d_step_cur = 0

        self.replay_buffer = ReplayBuffer(1000, replace_probability=0.2)
        
        self.loss_D = nn.BCEWithLogitsLoss()
        # self.rec_loss = nn.L1Loss()
        self.rec_loss = nn.MSELoss()
        
        self.training_passes = [{'optimizer':'D',
                                 'networks':['D'],
                                 'compute':self.D_pass,
                                 'loss_name':'D'},
                                {'optimizer':'G',
                                 'networks':['E', 'G', 'Z'],
                                 'compute':self.G_pass,
                                 'loss_name':'G'}]
        
        self.discriminator = Discriminator(self.eval_D, discriminator_type='loss')
        
        

    def clean(self):
        del self.fake_z
        del self.rec
        del self.output
        del self.rec_z
        self.losses = {}
        
    def forward(self, data):
        input = data[self.input_dataset]
        output = data[self.output_dataset]
        if self.mask:
            mask = data[self.mask]
        else:
            mask = None
        z = data[self.latent_dataset]
        
        # TODO: Forward pass does not need to evaluate everything, depending on lambdas?
        self.fake_z = self.networks['E'](input, output * (mask if mask is not None else 1))
        self.rec = self.networks['G'](torch.cat((input, self.networks['Z'](self.fake_z)), dim=1))
        self.output = self.networks['G'](torch.cat((input, self.networks['Z'](z)), dim=1))
        self.rec_z = self.networks['E'](input, self.output * (mask if mask is not None else 1))
    
    def eval_D(self, z):
        return self.networks['D'](z)

    
    def apply(self, data, datasets=None):
        # print('Requested', datasets)
        if datasets is None:
            datasets = [self.output_dataset]
            
        d = {}
        self.set_eval_mode()
        with torch.no_grad():
            if self.output_dataset in datasets:
                input = data[self.input_dataset]
                z = data[self.latent_dataset]
    
                out = self.networks['G'](torch.cat((input, self.networks['Z'](z)), dim=1))

                if isinstance(datasets, dict):
                    d[datasets[self.output_dataset]] = out
                else:
                    d[self.output_dataset] = out
            if 'Z_real' in datasets:
                input = data[self.input_dataset]
                output = data[self.output_dataset]
                if self.mask:
                    mask = data[self.mask]
                else:
                    mask = None
                z = self.networks['E'](input, output * (mask if mask is not None else 1))
                
                if isinstance(datasets, dict):
                    d[datasets['Z_real']] = z
                else:
                    d['Z_real'] = z
            
        return d
                
    def compute_D_losses(self, data):
        z = data[self.latent_dataset]
        fake_z = self.replay_buffer.query(self.fake_z.detach())
        
        self.discriminator.record_loss(self.losses, 'D', 'D_D', self.parameters['lambda_D_D'], real=z, fake=fake_z)
        self.discriminator.record_gradient_penalty_loss(self.losses, 'D', 'D_GP', self.parameters['lambda_GP'], real=z, fake=self.fake_z)

    
    def G_pass(self, data):
        output = data[self.output_dataset]
        if self.mask:
            mask = data[self.mask]
        else:
            mask = None
        z = data[self.latent_dataset]

        self.discriminator.record_generator_loss(self.losses, 'G', 'G_D', self.parameters['lambda_G_D'], self.fake_z)
        
        if self.parameters['lambda_rec'] > 0:
            self.record_loss('G', 'G_rec', self.parameters['lambda_rec'], self.rec_loss(self.rec * (mask if mask is not None else 1), output * (mask if mask is not None else 1)))
        if self.parameters['lambda_Z_rec'] > 0:
            self.record_loss('G', 'G_Z_rec', self.parameters['lambda_Z_rec'], self.rec_loss(self.rec_z, z))

    
    def D_pass(self, data):
        # Calculate grads for G pass (single forward call for both passes)
        self.set_requires_grad(['E', 'Z', 'G'])
        
        self.forward(data)        
        self.compute_D_losses(data)
    
                   
    def get_loss_names(self):
        return ['D', 'D_D', 'D_GP', 'G', 'G_rec', 'G_Z_rec', 'G_D']

