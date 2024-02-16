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

dirname = './results/csm_aae_20s'

parameters = yaml_load_many([f'{dirname}/fastmri_data_csm_lr.txt'])
parameters['datasets']['train']['type'] = 'DynamicDataset'
parameters['datasets']['train']['n_data'] = 1
parameters['datasets']['train']['indices'] = [['40']]

dataset = _dataset.create({x:y for x,y in parameters['datasets']['train'].items() if x not in ['training_only']})
dataset.load()

p = Path(f'{dirname}/fold1')

if (p / 'final.yaml').exists():
    model = load_from_file(p / 'final')
else:
    tmp = set((p / 'weights').glob('epoch*.yaml'))
    tmp2 = set((p / 'weights').glob('epoch_*_*.yaml'))
    tmp = list(tmp - tmp2)[0]
    model = load_from_file(tmp.parent / tmp.name[:-5])
model.to(device)

parameters = yaml.safe_load(open(f'{dirname}/fold1/parameters.yaml', 'r'))
model_runner = _model_runner.create(parameters['model_runners']['default'], model=model)

#%%
model_runner.input_dataset = ['in', 'noise']
del model_runner.processors[-1]
# del model_runner_phase.processors[2]

#%%



#%%
from matplotlib.widgets import Slider, Button

d,a = dataset.indexed_item(0)


z = model.apply({'in': torch.from_numpy(d['in'][:,0][None]).to(device), 'out': (torch.from_numpy(d['out'][:,0][None]).to(device))}, datasets=['Z_real'])

# nz = model.network_Z.model[0].in_features
nz = 4
d['noise'] = torch.zeros((nz,d['in'].shape[1]), device=device)
for i in range(d['noise'].shape[1]):
    d['noise'][:,i] = z['Z_real'][0,:]
    
sub = (slice(None),slice(0,1),slice(None),slice(None))
out = model_runner(d,a,seed=0,subimage=sub)

out = out['out']

im = tensor2grid(out[:,0].cpu().numpy(), vmax=1)



fig = plt.figure()
p = plt.imshow(im)

sliders = []

def update(val):
    d,a = dataset.indexed_item(sliders[-2].val)
    d['noise'] = torch.zeros((nz,d['in'].shape[1]), device=device)
    sliders[-1].valmax = d['in'].shape[1]-1
    sliders[-1].ax.set_xlim([0,d['in'].shape[1]-1])
    if sliders[-1].val > sliders[-1].valmax:
        sliders[-1].val = sliders[-1].valmax

    for i in range(nz):
        d['noise'][i] = sliders[i].val
    sub = (slice(None),slice(sliders[-1].val,sliders[-1].val+1),slice(None),slice(None))

    out = model_runner(d,a,seed=0,subimage=sub)
    out = out['out']
    im = tensor2grid(out[:,0].cpu().numpy(), vmax=1)

    p.set_data(im)
    fig.canvas.draw_idle()

def set_sliders(val):
    d,a = dataset.indexed_item(sliders[-2].val)
    S = sliders[-1].val

    z = model.apply({'in': torch.from_numpy(d['in'][:,S][None]).to(device), 'out': (torch.from_numpy(d['out'][:,S][None]).to(device))}, datasets=['Z_real'])

    for i in range(nz):
        sliders[i].valinit = z['Z_real'][0,i].item()
        sliders[i].vline.set_data(z['Z_real'][0,i].item(), [0,1])
        sliders[i].set_val(z['Z_real'][0,i].item())
    
    update(0)


for i in range(nz):
    sliders.append(Slider(
        ax=plt.axes([0.25, nz*0.01-i*0.01, 0.65, 0.01]),
        label=f'Z[{i}]',
        valmin=-3,
        valmax=3,
        valinit=z['Z_real'][0,i].item()))
    sliders[-1].on_changed(update)

sliders.append(Slider(
        ax=plt.axes([0.25, 0.95, 0.65, 0.03]),
        label='Image',
        valmin=0,
        valmax=dataset.number_of_items()-1,
        valinit=0,
        valstep=1))
sliders[-1].on_changed(update)

sliders.append(Slider(
        ax=plt.axes([0.25, 0.9, 0.65, 0.03]),
        label='Slice',
        valmin=0,
        valmax=d['in'].shape[1]-1,
        valinit=0,
        valstep=1))
sliders[-1].on_changed(update)



def show_real(x):
    pass

def show_gen(x):
    pass

def show_phase(x):
    pass

def show_csm(x):
    pass

def show_raw(x):
    pass

b1 = Button(ax=plt.axes([0,0,0.2,0.1]), label='Fit latent')
b1.on_clicked(set_sliders)

b2 = Button(ax=plt.axes([0,0.2,0.2,0.1]), label='Show real')
b2.on_clicked(show_real)

b3 = Button(ax=plt.axes([0,0.1,0.2,0.1]), label='Show gen')
b3.on_clicked(show_gen)

b4 = Button(ax=plt.axes([0,0.1,0.2,0.1]), label='Show phase')
b4.on_clicked(show_raw)

b5 = Button(ax=plt.axes([0,0.1,0.2,0.1]), label='Show CSM')
b5.on_clicked(show_csm)

b6 = Button(ax=plt.axes([0,0.1,0.2,0.1]), label='Show raw')
b6.on_clicked(show_raw)




