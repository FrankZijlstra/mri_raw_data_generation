import torch
import yaml

from . import model

@model('LoadFromFile', save_parameters=False)
def load_from_file(filename, load_optimizer=False):
    parameters = yaml.safe_load(open(str(filename) + '.yaml', 'r'))
    m = model.create(parameters)
    m.load(filename, load_optimizer=load_optimizer)
    
    return m
