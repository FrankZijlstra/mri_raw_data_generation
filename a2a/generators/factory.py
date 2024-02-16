from a2a.utils.factory import Factory, apply_on_all, factories
from a2a.utils.globals import patch_global_vars

from .background import BackgroundGenerator, BackgroundGeneratorPool

processor = Factory('processor')

class GeneratorFactory(Factory):
    def create(self, parameters, *args, **kwargs):
        if isinstance(parameters, str):
            parameters = {'type':parameters}

        if parameters['type'] in self.types:
            p = dict(parameters)
            name = p.pop('type')
            p = patch_global_vars(p)
                        
            if 'threads' in p:
                threads = p.pop('threads')
            else:
                # By default use the BackgroundGenerator
                threads = 1
            
            # TODO: Is there a nicer way to handle this?
            if threads > 1:
                ps = []
                for i in range(threads):
                    ps.append(dict(p))
                    for condition,action,pass_kwargs in self.parameter_handlers + [factories[x].parameter_handler() for x in factories if x not in self.ignore_default_handlers]:
                        for x in ps[i]:
                            if condition(x, ps[i], self.type_parameters[name]):
                                if len(pass_kwargs)>0:
                                    tp = dict(self.type_parameters[name])
                                    for arg in pass_kwargs:
                                        if arg in tp:
                                            tp[arg] = kwargs[arg]
                                else:
                                    tp = self.type_parameters[name]
                                ps[i][x] = action(x, ps[i], tp)
            else:
                for condition,action,pass_kwargs in self.parameter_handlers + [factories[x].parameter_handler() for x in factories if x not in self.ignore_default_handlers]:
                    for x in p:
                        if condition(x, p, self.type_parameters[name]):
                            if len(pass_kwargs)>0:
                                tp = dict(self.type_parameters[name])
                                for arg in pass_kwargs:
                                    if arg in tp:
                                        tp[arg] = kwargs[arg]
                            else:
                                tp = self.type_parameters[name]
                            p[x] = action(x, p, tp)

            for x in kwargs:
                if x not in p and x not in self.consume_kwargs:
                    p[x] = kwargs[x]

            if threads == 0:
                return self.types[name](*args, **p)
            elif threads == 1:
                return BackgroundGenerator(self.types[name](*args, **p))
            else:
                return BackgroundGeneratorPool([self.types[name](*args, **ps[x]) for x in range(threads)])
        else:
            raise ValueError(f'Unknown {self.name} type: {parameters["type"]}')

generator = GeneratorFactory('generator')
