import argparse
import os
from pathlib import Path

import numpy as np
import random
import shutil
import time
import datetime
import yaml
import sys
import math

import torch

from a2a.utils.yaml import yaml_load_many, yaml_dependencies
from a2a.utils.globals import set_global_var, set_global_vars, set_global, patch_global_vars, clear_global_vars

from a2a.data import dataset as _dataset
from a2a.generators import generator as _generator
from a2a.model_runners import model_runner as _model_runner

from a2a.models import model as _model
from a2a.trainers import trainer as _trainer



def load_datasets(parameters, fold, inference=False, dataset_names=None, datasets=None):
    if datasets is None:
        datasets = {}
        
    for dataset_name in parameters['datasets']:
        print(dataset_name)
        
        if dataset_names is not None and dataset_name not in dataset_names:
            continue
        
        if not inference and 'inference_only' in parameters['datasets'][dataset_name]:
            if parameters['datasets'][dataset_name]['inference_only']:
                continue
        
        if inference and 'training_only' in parameters['datasets'][dataset_name]:
            if parameters['datasets'][dataset_name]['training_only']:
                continue

        datasets[dataset_name] = _dataset.create({x:y for x,y in parameters['datasets'][dataset_name].items() if x not in ['training_only', 'inference_only']}, fold=fold)
        datasets[dataset_name].load()
    return datasets

def load_generators(parameters, generators=None, generator_names=None):
    if generators is None:
        generators = {}
        
    for generator_name in parameters['generators']:
        if generator_names is not None and generator_name not in generator_names:
            continue
        generators[generator_name] = _generator.create(parameters['generators'][generator_name])
        
    return generators

def load_model(parameters, device=None):
    model = _model.create(parameters['model'])
    
    if device is not None:
        model.to(device)
        
    return model

def load_model_runners(parameters, model, model_runner_names=None, model_runners=None):
    if model_runners is None:
        model_runners = {}
        
    for model_runner_name in parameters['model_runners']:
        if model_runner_names is not None and model_runner_name not in model_runner_names:
            continue
        
        model_runners[model_runner_name] = _model_runner.create(parameters['model_runners'][model_runner_name], model=model)
    
    return model_runners

def load_trainer(parameters, model):
    return _trainer.create(parameters['trainer'], model=model)

def run(parameter_files, output_dir, device='cuda:0', globals={}, seed=None):
    for parameter_file in parameter_files:
        if not Path(parameter_file).is_file():
            raise ValueError('Parameter file "{}" does not exist'.format(parameter_file))

    #%% Settings
    output_dir = Path(output_dir).resolve()
    parameters = yaml_load_many(parameter_files)
    
    if 'folds' not in parameters:
        parameters['folds'] = 1
    
    if 'model_runners' not in parameters:
        parameters['model_runners'] = {'default':'Default'}
    
    if 'globals' not in parameters:
        parameters['globals'] = {}

    #%% Create output directory and copy source
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_dir / 'source'):
        shutil.rmtree(output_dir / 'source')
    
    # Copy a2a source
    # TODO: Should also copy application-specific source code
    shutil.copytree(Path(__file__).absolute().parent, output_dir / 'source')
    
    # Copy parameter files
    for parameter_file in yaml_dependencies(parameter_files):
        shutil.copy(parameter_file, output_dir)
    
    with open(output_dir / 'commandline.txt', 'w') as fp:
        fp.write(' '.join(sys.argv))
    
    yaml.dump({'parameter_files':parameter_files,
               'output_dir':output_dir,
               'device':device,
               'globals':globals,
               'seed':seed}, open(output_dir / 'run_args.yaml', 'w'))
    
    #%%
    folds = parameters['folds']
    
    for fold in range(folds):
        log_dir = output_dir / f'fold{fold+1}'
    
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(log_dir / 'weights', exist_ok=True)       
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        else:
            np.random.seed(round(time.time()*1000) % (2**32 - 1))
            torch.manual_seed(round(time.time()*1000) % (2**32 - 1))
            random.seed(round(time.time()*1000) % (2**32 - 1))
            
            #TODO: use utils.set_seed
    
        #%%
        set_global_vars(**parameters['globals'])
        set_global('device', device)
        set_global('log_directory', log_dir)
        set_global('math', math) # TODO: Rework globals computation somehow
        
        for name, value in globals:
            set_global_var(name, value)
        
        #%% Load datasets
        datasets = {}
        set_global('datasets', datasets)
        
        load_datasets(parameters, fold, datasets=datasets)
        
        print('Datasets:')
        for dataset_name in datasets:
            try:
                print(f'  {dataset_name}: {datasets[dataset_name].number_of_items()}')
            except NotImplementedError:
                pass
    
        #%% Create generators
        generators = {}
        set_global('generators', generators)
        
        load_generators(parameters, generators=generators)

        #%% Create pytorch model
        model = load_model(parameters, device=device)
    
        for name,network in model.__dict__.items():
            if isinstance(network,torch.nn.Module):
                total_params = sum(p.numel() for p in network.parameters())
                trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
                if total_params > 0:
                    print(f'{name} parameters: {total_params}')
                    print(f'{name} trainable parameters: {trainable_params}')
                    
        set_global('model', model)
    
        #%% Create model runner
        model_runners = {}
        set_global('model_runners', model_runners)
        
        load_model_runners(parameters, model, model_runners=model_runners)
    
        #%% Store all necessary parameters for inference
        yaml.dump({'datasets':patch_global_vars(parameters['datasets']),
                   'model':patch_global_vars(parameters['model']),
                   'model_runners':patch_global_vars(parameters['model_runners']),
                   'stored_parameters':{x:datasets[x].stored_parameters for x in datasets if hasattr(datasets[x], 'stored_parameters')}
                   }, open(log_dir / 'parameters.yaml', 'w'))
    
        #%% Train model
        print('Training...')
        trainer = load_trainer(parameters, model)
        set_global('trainer', trainer)
        
        # TODO: Options for continuing interrupted training?
        # model.load(log_dir / 'final')
    
        trainer.train()
        model.save(log_dir / 'final')
    
        #%% Clean up for next fold
        clear_global_vars()
        del model, trainer, network
        del model_runners, datasets, generators
