from .preprocessor_base import register_op_preprocessor

from a2a.processors.gpu import GPU, CPU

register_op_preprocessor('GPU', GPU)
register_op_preprocessor('CPU', CPU)
