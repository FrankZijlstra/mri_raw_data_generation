import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-p', dest='parameter_files', type=str, nargs=argparse.ONE_OR_MORE, required=True, help='Filename of the parameter file')
parser.add_argument('-o', dest='output_dir', type=str, required=True, help='Output directory name')

parser.add_argument('-gpu', dest='gpu', type=int, default=0, help='ID of the GPU to run on')
parser.add_argument('-seed', dest='seed', type=int, help='Random seed')
parser.add_argument('-threads', dest='threads', type=int, default=4, help='Maximum number of threads for numpy/scipy/etc')

parser.add_argument('-g', dest='globals', type=str, action='append', nargs=2, metavar=('name', 'value'), help='Global(s)')

args = parser.parse_args()

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

import a2a
import generation_utils
import reconstruction_utils
# with torch.autograd.profiler.profile(use_cuda=True) as prof:
# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#     ], with_stack=True
# ) as prof:
#     a2a.run(args.parameter_files, args.output_dir, device=device, seed=args.seed)

# print(prof.key_averages().table(sort_by="cpu_time_total"))
import math

a2a.run(args.parameter_files, args.output_dir, device=device, seed=args.seed, globals=args.globals if args.globals else {})
