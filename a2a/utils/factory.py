from .globals import patch_global_vars

factories = {}

def apply_on_all(value, f):
    if isinstance(value, list):
        return [apply_on_all(x, f) for x in value]
    elif isinstance(value, dict):
        if 'type' in value:
            return f(value)
        else:
            return {x:apply_on_all(value[x], f) for x in value}
    else:
        return f(value)

class Factory:
    def __init__(self, name, ignore_default_handlers=[]):
        self.name = name
        self.types = {}
        self.save_parameters = {}
        self.type_parameters = {}
        self.parameter_handlers = []
        self.consume_kwargs = []
        self.ignore_default_handlers = ignore_default_handlers
        
        factories[name] = self
    
    def __call__(self, name, save_parameters=True, **kwargs):
        # print('Register', name)
        self.type_parameters[name] = {}
        for x,value in kwargs.items():
            self.type_parameters[name][x] = value
        
        def f(c):
            self.types[name] = c
            self.save_parameters[name] = save_parameters
            return c
        return f
    
    def parameter_handler(self):
        return (lambda x,p,tp: self.name + '_parameters' in tp and x in tp[self.name + '_parameters'], lambda x,p,tp: apply_on_all(p[x], self.create), [])

    # condition: function that takes parameter_name,parameters,type_parameters, returns boolean that decides whether to run action
    # action: function that takes parameter_name,parameters,type_parameters, performs action, result gets placed in parameters[parameter_name]
    #         if result is None, parameters[parameter_name] is removed
    def register_parameter_handler(self, condition, action, pass_kwargs=[]):
        self.parameter_handlers.append((condition, action, pass_kwargs))
    
    def register_simple_parameter_handler(self, name, action):
        self.register_parameter_handler(lambda x,p,tp: name in tp and x in tp[name], lambda x,p,tp: action(p[x]))
    
    def consume_kwarg(self, name):
        self.consume_kwargs.append(name)
    
    def create(self, parameters, *args, **kwargs):
        if isinstance(parameters, str):
            parameters = {'type':parameters}

        if parameters['type'] in self.types:
            p = dict(parameters)
            p = patch_global_vars(p)
            
            p_copy = dict(p)

            name = p.pop('type')
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
            
            try:
                ret = self.types[name](*args, **p)
                if self.save_parameters[name]:
                    ret._factory_parameters = p_copy
                return ret
            except:
                print(f'Exception in {name}:')
                raise
        else:
            raise ValueError(f'Unknown {self.name} type: {parameters["type"]}')


class FunctionFactory(Factory):
    def create(self, parameters, *args, **kwargs):
        if isinstance(parameters, str):
            parameters = {'type':parameters}

        if parameters['type'] in self.types:
            p = dict(parameters)
            name = p.pop('type')
            p = patch_global_vars(p)
            
            for condition,action,pass_kwargs in self.parameter_handlers + [factories[x].parameter_handler() for x in factories if x not in self.ignore_default_handlers]:
                for x in p:
                    if condition(x, p, self.type_parameters[name]):
                        if len(pass_kwargs)>0:
                            tp = dict(self.type_parameters[name])
                            for arg in pass_kwargs:
                                tp[arg] = kwargs[arg]
                        else:
                            tp = self.type_parameters[name]
                        p[x] = action(x, p, tp)

            for x in kwargs:
                if x not in p and x not in self.consume_kwargs:
                    p[x] = kwargs[x]
            
            def function(*a, **kw):
                return self.types[name](*a, *args, **p, **kw)
            
            return function
        else:
            raise ValueError(f'Unknown {self.name} type: {parameters["type"]}')