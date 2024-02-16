import h5py
from pathlib import Path
import sys
import lxml
import lxml.etree

from copy import copy

# Point to directory where your fastmri train and validation data is
traindir = Path('./fastmri/brain/multicoil_train')
valdir = Path('./fastmri/brain/multicoil_val')

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

# Set match to empty dictionary to match every scan and see full range of parameters to filter on
# match = {}

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

# Set coils to None to not select for coil and see all elements present in the matched scans
# coils = None

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

parameter_range = {}

for x in parameters:
    if all(k in x and x[k] == match[k] for k in match):
        files_filtered.append(x['filename'])
        
        for k in x:
            if k == 'filename':
                continue
            
            if k not in parameter_range:
                parameter_range[k] = set([x[k]])
            
            parameter_range[k] |= set([x[k]])

print('Parameter ranges:')
for k,v in parameter_range.items():
    print(f'  {k}:', sorted(list(v)))


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


#%% Count frequency of coil elements
coil_frequencies = {}

for i,file in enumerate(files_filtered2):
    # print(file)

    # Read data
    with h5py.File(file, 'r') as f:
        root = lxml.etree.fromstring(f['ismrmrd_header'][()])
        for elem in root.getiterator():
            elem.tag = lxml.etree.QName(elem).localname

    d = dictify(root)
    if 'coilLabel' in d['ismrmrdHeader']['acquisitionSystemInformation']:
        coilLabel = d['ismrmrdHeader']['acquisitionSystemInformation']['coilLabel']
        
        for x in coilLabel:
            name = x['coilName']
            if name not in coil_frequencies:
                coil_frequencies[name] = 0 
            coil_frequencies[name] += 1
    
print(f'Total files matched: {len(files_filtered2)}')
print('Coil element counts:')
for k,v in sorted(coil_frequencies.items()):
    print(' ', k, v)
