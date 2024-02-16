from .processor import register_op_processor

from a2a.processors.gpu import GPU, CPU

register_op_processor('GPU', GPU)
register_op_processor('CPU', CPU)
