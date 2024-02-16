from .factory import generator
import random

@generator('Generator', processor_parameters=['processors'])
def generator_func (processors=[]):
    while True:
        data = {}
        attr = {}
        for processor in processors:
            processor(data, attr)

        yield data


@generator('MultiGenerator', generator_parameters=['generators'])
def multi_generator (generators=[], weights=None):
    if weights is None:
        while True:
            for g in generators:
                yield next(g)
    else:
        while True:
            g = random.choices(generators, weights=weights)[0]
            yield next(g)

# TODO: MultiGenerator with threads? generators will be shared among instances?