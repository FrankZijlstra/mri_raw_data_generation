import h5py
import numpy as np
from pathlib import Path
import os
import sys
import lxml
import lxml.etree

from copy import copy
from coils import calculate_csm_inati_iter, ifft

# Point to directory where your fastmri data is
traindir = Path('./fastmri/brain/multicoil_train')
valdir = Path('./fastmri/brain/multicoil_val')

# Directory to write the dataset to
output_dir = Path('./data/fastmri_t1post')

# Describe scan type we want to filter
match = {'scantype': 'T1POST',
   'shape_phase': 320,
   'shape_read': 640,
   'B0': 2.8936,
   'TR': '250',
   'TE': '2.64',
   'TI': '300',
   'flipAngle_deg': '70',
   'sequence_type': 'Flash'}

# Coil elements to use
coils = ['HeadNeck_20:1:H11',
         'HeadNeck_20:1:H12',
         'HeadNeck_20:1:H13',
         'HeadNeck_20:1:H14',
         'HeadNeck_20:1:H21',
         'HeadNeck_20:1:H22',
         'HeadNeck_20:1:H23',
         'HeadNeck_20:1:H24',
         'HeadNeck_20:1:H31',
         'HeadNeck_20:1:H32',
         'HeadNeck_20:1:H33',
         'HeadNeck_20:1:H34',
         'HeadNeck_20:1:H41',
         'HeadNeck_20:1:H42',
         'HeadNeck_20:1:H43',
         'HeadNeck_20:1:H44']

# Multiply k-space data with this constant so the reconstructed data ends up with an intensity around 1
kspace_scale = 1e6

#%%
os.makedirs(output_dir, exist_ok=True)

#%%
def dictify(r,root=True):
    if root:
        return {r.tag : dictify(r, False)}
    d=copy(r.attrib)
    if r.text and r.text.strip() != '':
        d["_text"] = r.text
    for x in r.findall("./*"):
        if x.tag not in d:
            d[x.tag] = []
        d[x.tag].append(dictify(x,False))
        
    for x in d:
        if isinstance(d[x], list) and len(d[x]) == 1:
            d[x] = d[x][0]
    
    if len(list(d.keys())) == 1 and list(d.keys())[0] == '_text':
        return d['_text']
    return d


#%% Iterate through all fastMRI file and get relevant scan parameters
parameters = []

files = sorted(list(traindir.glob('*.h5')) + list(valdir.glob('*.h5')))

from tqdm.auto import tqdm
for file in tqdm(files, file=sys.stdout):
    
    pars = {}
    pars['filename'] = file
    
    if str(file).find('AXFLAIR') != -1:
        pars['scantype'] = 'FLAIR'
    elif str(file).find('AXT1POST') != -1:
        pars['scantype'] = 'T1POST'
    # # Uncomment to put T1PRE scans into a separate set (otherwise it's combined with AXT1)
    # elif str(file).find('AXT1PRE') != -1:
    #     pars['scantype'] = 'T1PRE'
    elif str(file).find('AXT1') != -1:
        pars['scantype'] = 'T1'
    elif str(file).find('AXT2') != -1:
        pars['scantype'] = 'T2'
    else:
        raise RuntimeError('Unknown scan type')
    
    with h5py.File(file, 'r') as f:
        root = lxml.etree.fromstring(f['ismrmrd_header'][()])
        for elem in root.getiterator():
            elem.tag = lxml.etree.QName(elem).localname

        pars['shape'] = f['kspace'].shape # slice, coils,read, phase
        pars['shape_slice'] = f['kspace'].shape[0]
        pars['shape_phase'] = f['kspace'].shape[3]
        pars['shape_read'] = f['kspace'].shape[2]

    d = dictify(root)
    pars['B0'] = float(d['ismrmrdHeader']['acquisitionSystemInformation']['systemFieldStrength_T'])

    if any(x in pars for x in d['ismrmrdHeader']['sequenceParameters']):
        raise RuntimeError('Double parameter')
    
    for x in d['ismrmrdHeader']['sequenceParameters']:
        pars[x] = d['ismrmrdHeader']['sequenceParameters'][x]
    
    # Skip data with less than 8 receive channels
    if int(d['ismrmrdHeader']['acquisitionSystemInformation']['receiverChannels']) < 8:
        continue
    
    parameters.append(pars)
    

#%% Select files that match the requested scan parameters
files_filtered = []

for x in parameters:
    if all(k in x and x[k] == match[k] for k in match):
        files_filtered.append(x['filename'])

#%% Select files that include the requested coils (if specified)
files_filtered2 = []

for file in files_filtered:
    with h5py.File(file, 'r') as f:
        root = lxml.etree.fromstring(f['ismrmrd_header'][()])
        for elem in root.getiterator():
            elem.tag = lxml.etree.QName(elem).localname

    d = dictify(root)
    if 'coilLabel' in d['ismrmrdHeader']['acquisitionSystemInformation']:
        coilLabel = d['ismrmrdHeader']['acquisitionSystemInformation']['coilLabel'] 
        c = [x['coilName'] for x in coilLabel]
    else:
        c = []
    
    # Only include files that include the requested coil elements
    if coils is None or all([x in c for x in coils]):
        files_filtered2.append(file)

#%% Reconstruct raw data and write data files
for i,file in enumerate(files_filtered2):
    print(i+1, file)

    # Read data
    with h5py.File(file, 'r') as f:
        root = lxml.etree.fromstring(f['ismrmrd_header'][()])
        for elem in root.getiterator():
            elem.tag = lxml.etree.QName(elem).localname

        k = np.array(f['kspace'])*kspace_scale
    
    # Rearrange and inverse FFT
    k = k.transpose(1,0,2,3)
    im = ifft(k)
    
    # Remove oversampling
    im = im[:,:,k.shape[2]//4:-k.shape[2]//4]
    
    # Select requested coils
    if coils is not None:
        d = dictify(root)
        coilLabel = d['ismrmrdHeader']['acquisitionSystemInformation']['coilLabel']
        c = [x['coilName'] for x in coilLabel]
        coilid = {x:i for i,x in enumerate(c)}
        p = [coilid[x] for x in coils]
        
        im = im[p]
    
    # Calculate coil sensititivies and reconstruct image
    csm, rec = calculate_csm_inati_iter(im, verbose=True, smoothing=5)
    
    # Normalize CSM phase with mean phase of the first coil element, correct reconstruction accordingly
    m = csm[0].mean()
    m /= abs(m)
    csm /= m
    rec *= m
    
    # Write to HDF5
    h5_file = output_dir / f'P{i+1}.h5'
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('/cc_im', data=im.astype(np.complex64), chunks=(1,1,im.shape[2],im.shape[3]), compression='gzip')
        f.create_dataset('/csm', data=csm.astype(np.complex64), chunks=(1,1,csm.shape[2],csm.shape[3]), compression='gzip')
        f.create_dataset('/rec', data=rec.astype(np.complex64), chunks=(1,rec.shape[1],rec.shape[2]), compression='gzip')


#%% Generate train/val/test split by 4 groups of scans (16/20 coils, skyra/prismafit scanner)
# Validation set is always the last scan from each the 4 groups
# Order is generated by mixing the groups proportionally
# Then, the test set is the last 20 scans from the order, the rest is the training set
# Note: The order in the experiment files is slightly different (generated with a slightly different implementation that was lost), though the sets are the same
# Note: Scan 4 and 116 were excluded
skyra = []
prismafit = []
coils20 = []
coils16 = []

for i,file in enumerate(files_filtered2):

    # Read data
    with h5py.File(file, 'r') as f:
        root = lxml.etree.fromstring(f['ismrmrd_header'][()])
        for elem in root.getiterator():
            elem.tag = lxml.etree.QName(elem).localname

    d = dictify(root)
    
    if d['ismrmrdHeader']['acquisitionSystemInformation']['systemModel'] == 'Skyra':
        skyra.append(i+1)
    else:
        prismafit.append(i+1)

    if d['ismrmrdHeader']['acquisitionSystemInformation']['receiverChannels'] == '20':
        coils20.append(i+1)
    else:
        coils16.append(i+1)

set1 = sorted(list(set(skyra) & set(coils16)))
set2 = sorted(list(set(skyra) & set(coils20) - set([4])))
set3 = sorted(list(set(prismafit) & set(coils16) - set([116])))
set4 = sorted(list(set(prismafit) & set(coils20)))

val = [f'{x}' for x in [set1[-1], set2[-1], set3[-1], set4[-1]]]

set1 = set1[:-1]
set2 = set2[:-1]
set3 = set3[:-1]
set4 = set4[:-1]

c1 = 0
c2 = 0
c3 = 0
c4 = 0

f1 = len(set1) / len(files_filtered2)
f2 = len(set2) / len(files_filtered2)
f3 = len(set3) / len(files_filtered2)
f4 = len(set4) / len(files_filtered2)

order = []
while set1 != [] or set2 != [] or set3 != [] or set4 != []:
    c1 += f1
    c2 += f2
    c3 += f3
    c4 += f4

    if c1 >= 1 and set1 != []:
        order.append(set1[0])
        set1 = set1[1:]
        c1 -= 1
    if c2 >= 1 and set2 != []:
        order.append(set2[0])
        set2 = set2[1:]
        c2 -= 1
    if c3 >= 1 and set3 != []:
        order.append(set3[0])
        set3 = set3[1:]
        c3 -= 1
    if c4 >= 1 and set4 != []:
        order.append(set4[0])
        set4 = set4[1:]
        c4 -= 1
        
train = [f'{x}' for x in order[:-20]]
test = [f'{x}' for x in order[-20:]]

print(f'ps_train: [{train}]')
print(f'ps_validation: [{val}]')
print(f'ps_test: [{test}]')
