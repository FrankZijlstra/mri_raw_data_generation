from a2a.utils.factory import Factory, apply_on_all

from a2a.generators import generator
from a2a.callbacks import callback


trainer = Factory('trainer')

from . import paired
