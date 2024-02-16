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

import torch

from a2a.utils.yaml import yaml_load_many, yaml_dependencies
from a2a.utils.globals import set_global_var, set_global_vars, set_global, patch_global_vars, clear_global_vars

from a2a.data import dataset as _dataset
from a2a.generators import generator as _generator
from a2a.model_runners import model_runner as _model_runner

from a2a.models import model as _model
from a2a.trainers import trainer as _trainer

def profile(parameter_files, device='cuda:0', globals={}, seed=None):
    for parameter_file in parameter_files:
        if not Path(parameter_file).is_file():
            raise ValueError('Parameter file "{}" does not exist'.format(parameter_file))

    #%% Settings
    output_dir = Path('./profiler_output').resolve()
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
    shutil.copytree(Path(__file__).absolute().parent, output_dir / 'source')
    
    # Copy parameter files
    for parameter_file in yaml_dependencies(parameter_files):
        shutil.copy(parameter_file, output_dir)
    
    with open(output_dir / 'commandline.txt', 'w') as fp:
        fp.write(' '.join(sys.argv))
    
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
    
        #%%
        set_global_vars(**parameters['globals'])
        set_global('device', device)
        set_global('log_directory', log_dir)
        
        for name, value in globals:
            set_global_var(name, value)
        
        #%% Load datasets
        datasets = {}
        set_global('datasets', datasets)
        
        for dataset_name in parameters['datasets']:
            print(dataset_name)
            
            if 'inference_only' in parameters['datasets'][dataset_name]:
                if parameters['datasets'][dataset_name]['inference_only']:
                    continue
    
            datasets[dataset_name] = _dataset.create({x:y for x,y in parameters['datasets'][dataset_name].items() if x not in ['training_only', 'inference_only']}, fold=fold)
            datasets[dataset_name].load()
        
        print('Datasets:')
        for dataset_name in datasets:
            try:
                print(f'  {dataset_name}: {datasets[dataset_name].number_of_items()}')
            except NotImplementedError:
                pass
    
        #%% Create generators
        generators = {}
        set_global('generators', generators)
        
        for generator_name in parameters['generators']:
            generators[generator_name] = _generator.create(parameters['generators'][generator_name])
    
        #%% Create pytorch model
        model = _model.create(parameters['model'])
        model.to(device)
    
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
        for model_runner_name in parameters['model_runners']:
            model_runners[model_runner_name] = _model_runner.create(parameters['model_runners'][model_runner_name], model=model)
        set_global('model_runners', model_runners)
    
        #%% Profile generators
        n_warmup = 100
        n_gen = 1000
        for name in generators:
            for i in range(n_warmup):
                d = next(generators[name])
            t0 = time.time()
            for i in range(n_gen):
                d = next(generators[name])
            torch.cuda.synchronize()
            t = time.time() - t0
            print(f'{name} generator sec/batch', t/n_gen)
            print(f'{name} generator batches/sec', n_gen/t)
        
        #%% Profile trainer
        def gen(d):
            while True:
                yield d
                
        trainer = _trainer.create(parameters['trainer'], model=model)
        
        trainer.epochs = 1
        t0 = time.time()
        trainer.train()
        t = time.time() - t0
        
        print('Training sec/epoch', t)
        print('Training batches/sec', trainer.batches_per_epoch/t)
        print('Training batches/sec (incl val)', (trainer.batches_per_epoch + trainer.validation_batches_per_epoch)/t)
    
        #%% Profile trainer
        
        trainer.epochs = 1
        trainer.callbacks = []
        t0 = time.time()
        trainer.train()
        t = time.time() - t0
        
        print('Training sec/epoch', t)
        print('Training batches/sec', trainer.batches_per_epoch/t)
        print('Training batches/sec (incl val)', (trainer.batches_per_epoch + trainer.validation_batches_per_epoch)/t)

