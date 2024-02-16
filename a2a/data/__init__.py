from a2a.utils.factory import Factory, apply_on_all

from a2a.preprocessors import preprocessor

dataloader = Factory('dataloader')
dataset = Factory('dataset')

from . import hdf5_dataloader

from . import dataset_base
from . import static_dataset
from . import dynamic_dataset
