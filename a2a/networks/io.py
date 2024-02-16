import torch
import yaml

from . import network

# TODO: A LoadFromFile network that's being trained should not be saved as a LoadFromFile network
@network('LoadFromFile', save_parameters=False)
def load_from_file(filename):
    parameters = yaml.safe_load(open(str(filename) + '.yaml', 'r'))
    net = network.create(parameters)
    net.load_state_dict(torch.load(str(filename) + '.pth'))
    
    return net

def save_to_file(filename, net):
    torch.save(net.state_dict(), str(filename) + '.pth')
    yaml.dump(net._factory_parameters, open(str(filename) + '.yaml', 'w'))
