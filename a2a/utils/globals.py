import math

def get_global_value(key):
    return patch_global_vars(f'%{key}%')

global_vars = {}
main_globals = {'get_global_value': get_global_value}

def set_global_var(name, value):
    global_vars[f'%{name}%'] = value

def set_global(name, value):
    main_globals[name] = value
    
def get_global(name):
    return main_globals[name]

def set_global_vars(**kwargs):
    for key,value in kwargs.items():
        global_vars[f'%{key}%'] = value

def clear_global_vars():
    global_vars.clear()
    main_globals.clear()
    main_globals['get_global_value'] = get_global_value

def patch_global_vars(x):
    if isinstance(x, str):
        if x in global_vars:
            v = str(global_vars[x])
            if len(v) >= 5 and v[:5] == 'eval ':
                return eval(v[5:], main_globals)
            else:
                return v
        
    if isinstance(x,dict):
        for d in x:
            x[d] = patch_global_vars(x[d])
    
    if isinstance(x,list):
        for i,v in enumerate(x):
            x[i] = patch_global_vars(v)
    
    if isinstance(x,tuple):
        return tuple([patch_global_vars(y) for y in x])
    
    return x
