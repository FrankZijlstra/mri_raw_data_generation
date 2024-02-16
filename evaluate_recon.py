#%% Parse command line
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-p', dest='parameter_files', type=str, nargs=argparse.ONE_OR_MORE, required=True, help='Filename of the parameter file')
parser.add_argument('-o', dest='output_dir', type=str, required=True, help='Output directory name')
parser.add_argument('-io', dest='inference_output_dir', type=str, default='inference', help='Output directory name')

parser.add_argument('-gpu', dest='gpu', type=int, default=0, help='ID of the GPU to run on')
parser.add_argument('-seed', dest='seed', type=int, help='Random seed')
parser.add_argument('-threads', dest='threads', type=int, default=4, help='Maximum number of threads for numpy/scipy/etc')

parser.add_argument('-g', dest='globals', type=str, action='append', nargs=2, metavar=('name', 'value'), help='Global(s)')

args = parser.parse_args()

if not args.output_dir:
    parser.error('Either -output_dir or -base_dir needs to be provided')

for parameter_file in args.parameter_files:
    if not Path(parameter_file).is_file():
        raise Exception('Parameter file "{}" does not exist'.format(parameter_file))
    
#%% pytorch init
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch

if not torch.cuda.is_available():
    raise Exception("No GPU found")

device = torch.device("cuda:0")
print('Device: {}'.format(torch.cuda.get_device_name(device.index)))

torch.backends.cudnn.benchmark = True

os.environ["OMP_NUM_THREADS"] = str(args.threads)

#%%
import numpy as np
import random
import shutil
import time
import datetime
import yaml
import sys

from pathlib import Path

import a2a
from a2a.utils.yaml import yaml_load_many
from a2a.utils.globals import set_global_var, set_global_vars, set_global, patch_global_vars, clear_global_vars

from a2a.data import dataset as _dataset
from a2a.generators import generator as _generator
from a2a.model_runners import model_runner as _model_runner

from a2a.models import model as _model
from a2a.trainers import trainer as _trainer

import reconstruction_utils

#%% Settings
output_dir = Path(args.output_dir).resolve()

parameters = yaml_load_many(args.parameter_files)

if 'folds' not in parameters:
    parameters['folds'] = 1

if 'model_runners' not in parameters:
    parameters['model_runners'] = {'default':'Default'}

if 'globals' not in parameters:
    parameters['globals'] = {}

#%% Create output directory and copy source


#%%
folds = parameters['folds']

for fold in range(folds):
    log_dir = output_dir / f'fold{fold+1}'
    inference_dir = log_dir / args.inference_output_dir
    os.makedirs(inference_dir, exist_ok=True)    
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    else:
        np.random.seed(round(time.time()*1000) % (2**32 - 1))
        torch.manual_seed(round(time.time()*1000) % (2**32 - 1))
        random.seed(round(time.time()*1000) % (2**32 - 1))


    #%%
    set_global_vars(**parameters['globals'])
    set_global('device', device)
    set_global('log_directory', log_dir)
    
    if args.globals is not None:
        for name, value in args.globals:
            set_global_var(name, value)
    
    #%% Load datasets
    datasets = {}
    set_global('datasets', datasets)
    
    stored_parameters = yaml.safe_load(open(log_dir / 'parameters.yaml', 'r'))
    for x in stored_parameters['stored_parameters']:
        datasets[x] = {'stored_parameters': stored_parameters['stored_parameters'][x]}
    
    for dataset_name in parameters['datasets']:
        print(dataset_name)
        
        if 'training_only' in parameters['datasets'][dataset_name]:
            if parameters['datasets'][dataset_name]['training_only']:
                continue

        datasets[dataset_name] = _dataset.create({x:y for x,y in parameters['datasets'][dataset_name].items() if x not in ['training_only', 'inference_only']}, fold=fold)
        datasets[dataset_name].load()
    
    print('Datasets:')
    for dataset_name in datasets:
        try:
            print(f'  {dataset_name}: {datasets[dataset_name].number_of_items()}')
        except NotImplementedError:
            pass
    
    #%% Create pytorch model
    model = _model.create(stored_parameters['model'])
    model.to(device)

    for name,network in model.__dict__.items():
        if isinstance(network,torch.nn.Module):
            total_params = sum(p.numel() for p in network.parameters())
            trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
            if total_params > 0:
                print(f'{name} parameters: {total_params}')
                print(f'{name} trainable parameters: {trainable_params}')

    model.load(log_dir / 'final')
    set_global('model', model)
    
    #%% Create model runner
    model_runners = {}
    for model_runner_name in stored_parameters['model_runners']:
        model_runners[model_runner_name] = _model_runner.create(stored_parameters['model_runners'][model_runner_name], model=model)
    set_global('model_runners', model_runners)
    
    #%% Apply model
    import matplotlib.pyplot as plt
    import nibabel as nib
    import scipy.ndimage as ndi
    
    from a2a.processors.metrics import SSIM2D, AbsoluteError, SquaredError
    
    test_dataset_name = 'test'
    model_runner_name = 'default'
    true_output_name = 'out'
    output_name = 'out'
    
    ae = AbsoluteError()
    se = SquaredError()
    ss = SSIM2D()
    
    with open(inference_dir / 'metrics.csv', "w") as f:
        f.write('P,MAE,MSE,RMSE,SSIM,SSIM2,masked_MAE,masked_MSE,masked_RMSE,masked_SSIM,premasked_SSIM,masked_SSIM2,premasked_SSIM2\n')
    
        # Remember to make dataset a DynamicDataset to load each item as it is needed
        for i in range(datasets[test_dataset_name].number_of_items()):
        # for i in [datasets[test_dataset_name].number_of_items()-1]:
            data = datasets[test_dataset_name].indexed_item(i)
            output = model_runners[model_runner_name](data[0], data[1])[output_name].cpu().numpy()[0]
            true_output = data[0]['out'][0]
            
            nib.save(nib.Nifti1Image(true_output.transpose(2,1,0), np.eye(4)), inference_dir / f'true_{i}.nii.gz')
            nib.save(nib.Nifti1Image(output.transpose(2,1,0), np.eye(4)), inference_dir / f'rec_{i}.nii.gz')
            
            mask = ndi.gaussian_filter(true_output, 3.0)>0.1
            nib.save(nib.Nifti1Image(np.uint8(mask.transpose(2,1,0)), np.eye(4)), inference_dir / f'mask_{i}.nii.gz')

            ae_map = ae(output, true_output)
            se_map = se(output, true_output)
            ssim_map = ss(output, true_output)
            ssimm_map = ss(output*mask, true_output*mask)
            
            ss2 = SSIM2D(data_range=true_output.max())
            ssim2_map = ss2(output, true_output)
            ssimm2_map = ss2(output*mask, true_output*mask)
            
            
            mae = ae_map.mean()
            masked_mae = ae_map[mask].mean()
            mse = se_map.mean()
            masked_mse = se_map[mask].mean()
            rmse = np.sqrt(se_map.mean())
            masked_rmse = np.sqrt(se_map[mask].mean())
            ssim = ssim_map.mean()
            masked_ssim = ssim_map[mask].mean()
            premasked_ssim = ssimm_map[mask].mean()
            ssim2 = ssim2_map.mean()
            masked_ssim2 = ssim2_map[mask].mean()
            premasked_ssim2 = ssimm2_map[mask].mean()
            f.write(f'{i},{mae},{mse},{rmse},{ssim},{ssim2},{masked_mae},{masked_mse},{masked_rmse},{masked_ssim},{premasked_ssim},{masked_ssim2},{premasked_ssim2}\n')

