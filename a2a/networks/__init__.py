from a2a.utils.factory import Factory, FunctionFactory

activation = FunctionFactory('activation')
normalization = FunctionFactory('normalization')
loss = Factory('loss')
network = Factory('network')


class OptimizerFactory(Factory):
    def create_optimizer(self, x, p, tp):
        tpp = tp[self.name + '_parameters'][x]
        
        if isinstance(tpp, list):
            pars = []
            
            # Optimizers is not a dictionary of optimizers
            for networks in tpp:
                if isinstance(networks, dict):
                    # Networks is a dictionary
                    for y, nets in networks.items():
                        for net in nets:
                            pars.extend(p[y][net].parameters())
                else:
                    pars.extend(p[networks].parameters())
                
                
            if isinstance(p[x], str):
                p[x] = {'type':p[x]}
            opt = optimizer.create(p[x], pars)
            return opt
        else:
            # Dictionary of optimizers
            opts = {}
            for optimizer_name, networks in tpp.items():
                pars = []
                
                for net in networks:
                    if isinstance(net, dict):
                        for y, nets in net.items():
                            for net in nets:
                                pars.extend(p[y][net].parameters())
                    else:
                        pars.extend(p[net].parameters())
                
                if isinstance(p[x][optimizer_name], str):
                    p[x][optimizer_name] = {'type':p[x][optimizer_name]}
                opts[optimizer_name] = optimizer.create(p[x][optimizer_name], pars)
            return opts
    
    def parameter_handler(self):
        return (lambda x, p, tp: self.name + '_parameters' in tp and x in tp[self.name + '_parameters'], self.create_optimizer, [])

optimizer = OptimizerFactory('optimizer')


from . import unet
from . import fcn
from . import layers
from . import utils
from . import mlp
from . import io
