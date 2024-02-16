#%% pytorch init
import torch

if not torch.cuda.is_available():
    raise Exception("No GPU found")

device = torch.device("cuda:0")
print('Device: {}'.format(torch.cuda.get_device_name(device.index)))

torch.backends.cudnn.benchmark = True

#%%
import yaml
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np

import a2a
from a2a.utils.yaml import yaml_load_many
from a2a.utils.globals import set_global

from a2a.data import dataset as _dataset
from a2a.model_runners import model_runner as _model_runner

from a2a.models.io import load_from_file
from a2a.callbacks.image_writers import tensor2grid

import generation_utils

#%% Settings
datasets = {}
set_global('datasets', datasets)

dirname_csm = './results/csm_aae_20s'
dirname_phase = './results/phase_aae_20s'

parameters = yaml_load_many([f'{dirname_csm}/fastmri_data_csm_lr.txt'])
parameters['datasets']['train']['type'] = 'DynamicDataset'
parameters['datasets']['train']['n_data'] = 1
parameters['datasets']['train']['indices'] = [['10', '50']]

parameters['datasets']['train']['dataset']['phase'] = dict(parameters['datasets']['train']['dataset']['out'])
parameters['datasets']['train']['dataset']['phase']['channels'] = ['/rec']

parameters['datasets']['train']['dataset']['raw'] = dict(parameters['datasets']['train']['dataset']['out'])
parameters['datasets']['train']['dataset']['raw']['channels'] = ['/cc_im']

parameters['preprocessing'][0]['dataset'] = ['in', 'out', 'phase']
parameters['preprocessing'][1]['dataset'] = ['in', 'out', 'phase']
parameters['preprocessing'][2]['dataset'] = ['in', 'out', 'phase']
parameters['preprocessing'][3]['dataset'] = ['out', 'phase', 'raw']

dataset = _dataset.create({x:y for x,y in parameters['datasets']['train'].items() if x not in ['training_only']})
dataset.load()

p = Path(f'{dirname_csm}/fold1')

model_csm = load_from_file(p / 'final')
model_csm.to(device)

parameters = yaml.safe_load(open(f'{dirname_csm}/fold1/parameters.yaml', 'r'))
model_runner_csm = _model_runner.create(parameters['model_runners']['default'], model=model_csm)


p = Path(f'{dirname_phase}/fold1')

model_phase = load_from_file(p / 'final')
model_phase.to(device)
model_phase.latent_dataset = 'noise_phase'

parameters = yaml.safe_load(open(f'{dirname_phase}/fold1/parameters.yaml', 'r'))
model_runner_phase = _model_runner.create(parameters['model_runners']['default'], model=model_phase)

#%%
model_runner_csm.input_dataset = ['in', 'noise']
del model_runner_csm.processors[-1]

model_runner_phase.input_dataset = ['in', 'noise_phase']
del model_runner_phase.processors[-1]


#%%
from matplotlib.widgets import Slider, Button

d,a = dataset.indexed_item(0)


z_csm = model_csm.apply({'in': torch.from_numpy(d['in'][:,0][None]).to(device),
                         'out': (torch.from_numpy(d['out'][:,0][None]).to(device))},
                        datasets=['Z_real'])
z_phase = model_phase.apply({'in': torch.from_numpy(d['in'][:,0][None]).to(device),
                             'out': (torch.from_numpy(d['phase'][:,0][None]).to(device))},
                            datasets=['Z_real'])


nz = 4
d['noise'] = torch.zeros((nz,d['in'].shape[1]), device=device)
for i in range(d['noise'].shape[1]):
    d['noise'][:,i] = z_csm['Z_real'][0,:]

d['noise_phase'] = torch.zeros((nz,d['in'].shape[1]), device=device)
for i in range(d['noise_phase'].shape[1]):
    d['noise_phase'][:,i] = z_phase['Z_real'][0,:]


sub = (slice(None),slice(0,1),slice(None),slice(None))
out_csm = model_runner_csm(d,a,seed=0,subimage=sub)['out']
out_phase = model_runner_phase(d,a,seed=0,subimage=sub)['out']

csm = tensor2grid(out_csm[:,0].cpu().numpy()[:,::-1], vmax=1)
phase = tensor2grid(out_phase[:,0].cpu().numpy()[:,::-1], vmax=1)

mode = 'phase'
truth = False

fig = plt.figure()
p = plt.imshow(phase)

sliders = []
sliders_phase = []

def update(val):
    global mode
    
    d,a = dataset.indexed_item(slider_dataset.val)
    d['noise'] = torch.zeros((nz,d['in'].shape[1]), device=device)
    d['noise_phase'] = torch.zeros((nz,d['in'].shape[1]), device=device)
    slider_slice.valmax = d['in'].shape[1]-1
    slider_slice.ax.set_xlim([0,d['in'].shape[1]-1])
    if slider_slice.val > slider_slice.valmax:
        slider_slice.val = slider_slice.valmax

    sub = (slice(None),slice(slider_slice.val,slider_slice.val+1),slice(None),slice(None))

    if truth or mode == 'magnitude':
        if mode == 'csm':
            csm = tensor2grid(d['out'][sub][:,0,::-1], vmax=1)
            p.set_data(csm)
        elif mode == 'phase':
            phase = tensor2grid(d['phase'][sub][:,0,::-1], vmax=1)
            p.set_data(phase)
        elif mode == 'magnitude':
            mag = tensor2grid(d['in'][sub][:,0,::-1], vmax=1, complex=False)
            p.set_data(mag)
        elif mode == 'raw':
            raw = tensor2grid(d['raw'][sub][:,0,::-1], vmax=0.5)
            p.set_data(raw)
        fig.canvas.draw_idle()
        return

    
    if mode == 'csm' or mode == 'raw':
        for i in range(nz):
            d['noise'][i] = sliders[i].val
        
        out_csm = model_runner_csm(d,a,seed=0,subimage=sub)['out']
        csm = tensor2grid(out_csm[:,0].cpu().numpy()[:,::-1], vmax=1)
    
        p.set_data(csm)
    
    if mode == 'phase' or mode == 'raw':
        for i in range(nz):
            d['noise_phase'][i] = sliders_phase[i].val
    
        out_phase = model_runner_phase(d,a,seed=0,subimage=sub)['out']
        phase = tensor2grid(out_phase[:,0].cpu().numpy()[:,::-1], vmax=1)
    
        p.set_data(phase)
        
    if mode == 'raw':    
        phase = out_phase[0] + 1j * out_phase[1]
        phase /= abs(phase)
        
        csm = out_csm[:16] + 1j * out_csm[16:]
        
        raw = d['in'][sub] * (phase * csm).cpu().numpy()
        raw = np.concatenate((raw.real, raw.imag), axis=0)
        raw = tensor2grid(raw[:,0,::-1], vmax=0.5)

        p.set_data(raw)
        
    fig.canvas.draw_idle()
        

def set_sliders(val):
    d,a = dataset.indexed_item(slider_dataset.val)
    S = slider_slice.val

    if mode == 'csm' or mode == 'raw':
        z_csm = model_csm.apply({'in': torch.from_numpy(d['in'][:,S][None]).to(device),
                                 'out': (torch.from_numpy(d['out'][:,S][None]).to(device))},
                                datasets=['Z_real'])
        
        for i in range(nz):
            sliders[i].valinit = z_csm['Z_real'][0,i].item()
            sliders[i].vline.set_data([z_csm['Z_real'][0,i].item()], [0,1])
            sliders[i].set_val(z_csm['Z_real'][0,i].item())
    if mode == 'phase' or mode == 'raw':
        z_phase = model_phase.apply({'in': torch.from_numpy(d['in'][:,S][None]).to(device),
                                     'out': (torch.from_numpy(d['phase'][:,S][None]).to(device))},
                                    datasets=['Z_real'])
        
        for i in range(nz):
            sliders_phase[i].valinit = z_phase['Z_real'][0,i].item()
            sliders_phase[i].vline.set_data([z_phase['Z_real'][0,i].item()], [0,1])
            sliders_phase[i].set_val(z_phase['Z_real'][0,i].item())
    
    update(0)


for i in range(nz):
    sliders.append(Slider(
        ax=plt.axes([0.25, nz*0.02-i*0.02, 0.65, 0.01]),
        label=f'csm[{i}]',
        valmin=-3,
        valmax=3,
        valinit=z_csm['Z_real'][0,i].item()))
    sliders[-1].on_changed(update)
    
for i in range(nz):
    sliders_phase.append(Slider(
        ax=plt.axes([0.25, nz*0.02-i*0.02 + 0.1, 0.65, 0.01]),
        label=f'phase[{i}]',
        valmin=-3,
        valmax=3,
        valinit=z_phase['Z_real'][0,i].item()))
    sliders_phase[-1].on_changed(update)


slider_dataset = Slider(
        ax=plt.axes([0.25, 0.95, 0.65, 0.03]),
        label='Image',
        valmin=0,
        valmax=dataset.number_of_items()-1,
        valinit=0,
        valstep=1)
slider_dataset.on_changed(update)

slider_slice = Slider(
        ax=plt.axes([0.25, 0.9, 0.65, 0.03]),
        label='Slice',
        valmin=0,
        valmax=d['in'].shape[1]-1,
        valinit=0,
        valstep=1)
slider_slice.on_changed(update)

def show_real(x):
    global truth
    truth = True
    
    update(0)

def show_gen(x):
    global truth
    truth = False
    
    update(0)

def show_phase(x):
    global mode

    mode = 'phase'
    update(0)

def show_csm(x):
    global mode

    mode = 'csm'
    update(0)

def show_raw(x):
    global mode

    mode = 'raw'
    update(0)

def show_magnitude(x):
    global mode

    mode = 'magnitude'
    update(0)

b1 = Button(ax=plt.axes([0,0,0.2,0.1]), label='Fit latent')
b1.on_clicked(set_sliders)

b2 = Button(ax=plt.axes([0,0.2,0.2,0.1]), label='Show real')
b2.on_clicked(show_real)

b3 = Button(ax=plt.axes([0,0.1,0.2,0.1]), label='Show gen')
b3.on_clicked(show_gen)

b4 = Button(ax=plt.axes([0,0.3,0.2,0.1]), label='Show phase')
b4.on_clicked(show_phase)

b5 = Button(ax=plt.axes([0,0.4,0.2,0.1]), label='Show CSM')
b5.on_clicked(show_csm)

b6 = Button(ax=plt.axes([0,0.5,0.2,0.1]), label='Show raw')
b6.on_clicked(show_raw)

b7 = Button(ax=plt.axes([0,0.6,0.2,0.1]), label='Show magnitude')
b7.on_clicked(show_magnitude)


plt.show()

