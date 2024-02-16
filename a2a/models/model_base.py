import torch
import yaml

from torch import autograd

import a2a.networks as networks

def get_item (x):
    if torch.is_tensor(x):
        return x.item()
    else:
        return x

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

class Model:
    def __init__(self, networks={}, optimizers={}, parameters=None):
        if not hasattr(self, 'parameters'):
            self.parameters = {}
        
        if parameters is not None:
            for x in parameters:
                self.parameters[x] = parameters[x]
        self.networks = networks
        self.optimizers = optimizers
        
        self.losses = {}
    
    def forward(self, input):
        pass
    
    def clean(self):
        self.losses = {}
    
    def set_train_mode(self):
        for x in self.networks.values():
            x.train()

    def set_eval_mode(self):
        for x in self.networks.values():
            x.eval()
    
    def train(self, data):
        self.set_train_mode()
        
        for p in self.training_passes:
            if 'optimizer' in p:
                self.optimizers[p['optimizer']].zero_grad()
            
            # Could also just iterate over optimizer parameters?
            
            for x in self.networks:
                for param in self.networks[x].parameters():
                    param.requires_grad = False
            for x in p['networks']:
                for param in self.networks[x].parameters():
                    param.requires_grad = True
            
            p['compute'](data)
            
            if 'loss_name' in p:
                self.losses[p['loss_name']].backward()
                
                # TODO: This prevents a pass from computing gradients for a future pass (e.g. discriminator pass in GAN does both D and G networks)
                if self.clip_gradient_norm > 0: # TODO: clip_gradient_norm not defined
                    for x in p['networks']:
                        torch.nn.utils.clip_grad_norm_(self.networks[x].parameters(), self.clip_gradient_norm)
                
                if 'optimizer' in p:
                    self.optimizers[p['optimizer']].step()

    
    def validate(self, data):
        self.set_eval_mode()
        
        for p in self.training_passes:                       
            for x in self.networks:
                for param in self.networks[x].parameters():
                    param.requires_grad = False
            
            p['compute'](data)
            

    def to(self, device):
        for x in self.networks:
            self.networks[x].to(device)
        for x in self.optimizers:
            optimizer_to(self.optimizers[x], device)
    
    def __call__(self, data, datasets=None):
        return self.apply(data, datasets=datasets)
    
    def apply(self, data, datasets=None):
        return {}
        
    def get_losses(self):
        return {x: get_item(self.losses[x]) for x in self.losses}
    
    def get_loss_names(self):
        return []
    
    def record_loss(self, loss_group_name, loss_name, loss_lambda, loss_value):
        self.losses[loss_name] = loss_value
        
        if loss_group_name is not None:
            if loss_group_name not in self.losses:
                self.losses[loss_group_name] = 0
            self.losses[loss_group_name] += loss_lambda * self.losses[loss_name]
        
    def get_device(self):
        if self.networks == {}:
            return None
        return next(self.networks[[x for x in self.networks][0]].parameters()).device
    
    def set_learning_rate(self, learning_rate):
        if isinstance(learning_rate, dict):
            for x in self.optimizers:
                if x in learning_rate:
                    for p in self.optimizers[x].param_groups:
                        p['lr'] = learning_rate[x]
        else:
            for x in self.optimizers:
                for p in self.optimizers[x].param_groups:
                    p['lr'] = learning_rate

    def set_parameters(self, parameters):
        for p in parameters:
            self.parameters[p] = parameters[p]
            
    def set_requires_grad(self, networks, value=True):
        for x in networks:
            for param in self.networks[x].parameters():
                param.requires_grad = value
    
    def load(self, base_filename, load_optimizer=False):
        for x in self.networks:
            #self.networks[x] = networks.io.load_from_file(str(base_filename) + f'_net{x}').to(self.get_device())
            self.networks[x].load_state_dict(torch.load(str(base_filename) + f'_net{x}.pth')) # networks.io.load_from_file(str(base_filename) + f'_net{x}').to(self.get_device())

        if load_optimizer:
            for x in self.optimizers:
                self.optimizers[x].load_state_dict(torch.load(str(base_filename) + f'_opt{x}.pth'))

    def save(self, base_filename, save_optimizer=False):
        yaml.dump(self._factory_parameters, open(str(base_filename) + '.yaml', 'w'))
        
        for x in self.networks:
            networks.io.save_to_file(str(base_filename) + f'_net{x}', self.networks[x])

        if save_optimizer:
            for x in self.optimizers:
                torch.save(self.optimizers[x].state_dict(), str(base_filename) + f'_opt{x}.pth')





class Discriminator:
    def __init__(self, D, discriminator_type='loss', loss=torch.nn.BCEWithLogitsLoss()):
        self.discriminator_type = discriminator_type
        self.loss = loss
        self.D = D
    
    def record_loss(self, losses, loss_group_name, loss_name, loss_lambda, real=None, fake=None):   
        if loss_lambda == 0:
            return
        
        if self.discriminator_type == 'wgan':
            losses[loss_name] = 0
            if real is not None:
                losses[f'{loss_name}_real'] = self.D(real).mean()
                losses[loss_name] += losses[f'{loss_name}_real']
            
            if fake is not None:
                losses[f'{loss_name}_fake'] = -self.D(fake.detach()).mean()
                losses[loss_name] += losses[f'{loss_name}_fake']
        else:
            losses[loss_name] = 0
            if real is not None:
                tmp = self.D(real)
                losses[f'{loss_name}_real'] = self.loss(tmp, torch.ones_like(tmp))
                losses[loss_name] += 0.5*losses[f'{loss_name}_real']
                
            if fake is not None:
                tmp = self.D(fake.detach())
                losses[f'{loss_name}_fake'] = self.loss(tmp, torch.zeros_like(tmp))
                losses[loss_name] += 0.5*losses[f'{loss_name}_fake']
        
        if loss_group_name not in losses:
            losses[loss_group_name] = 0
        losses[loss_group_name] += loss_lambda * losses[loss_name]
    
    def record_gradient_penalty_loss(self, losses, loss_group_name, loss_name, loss_lambda, real, fake):
        if loss_lambda == 0:
            return
        
        alpha = torch.rand((real.shape[0],) + (1,)*(real.ndim-1), device=real.device)

        interp = alpha * real + ((1 - alpha) * fake.detach())
        interp.requires_grad_(True)
        
        f_interp = self.D(interp)
    
        gradients = autograd.grad(
            outputs=f_interp,
            inputs=interp,
            grad_outputs=torch.ones(f_interp.shape, device=real.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        losses[loss_name] = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        if loss_group_name not in losses:
            losses[loss_group_name] = 0
        losses[loss_group_name] += loss_lambda * losses[loss_name]
    
    def record_generator_loss(self, losses, loss_group_name, loss_name, loss_lambda, fake):
        if loss_lambda == 0:
            return
        
        if self.discriminator_type == 'wgan':
            losses[loss_name] = self.D(fake).mean()
        else:
            tmp = self.D(fake)
            losses[loss_name] = self.loss(tmp, torch.ones_like(tmp))
        
        if loss_group_name not in losses:
            losses[loss_group_name] = 0
        losses[loss_group_name] += loss_lambda * losses[loss_name]
