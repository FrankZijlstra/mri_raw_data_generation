import yaml
from pathlib import Path

def update(x, y):
    # TODO: Can we search for any references to the field when overwriting?
    for k,v in y.items():
        if k in x and type(v) == str and v[0] == '%' and v[-1] == '%':
            x[k] = v       
        elif k in x and type(x[k]) == dict:
            update(x[k], v)
        elif k in x and type(x[k]) == list:
            x[k].clear()
            x[k].extend(v)
        else:
            x[k] = v

def yaml_load(filename):
    with open(filename, 'r') as f:
        d = {}
        while True:
            pos = f.tell()
            line = f.readline().strip()
            if line.startswith('include'):
                update(d, yaml_load(filename.parent / line[8:]))
            else:
                f.seek(pos)
                break
            
        update(d, yaml.safe_load(f))
        return d

def yaml_load_many(filenames):
    d = {}
    
    for f in filenames:
        y = yaml_load(Path(f))
        update(d, y)
    
    return d

def yaml_load_dependencies(filename):
    with open(filename, 'r') as f:
        d = set([filename])
        while True:
            line = f.readline().strip()
            if line.startswith('include'):
                d |= set([filename.parent / line[8:]])
                d |= yaml_load_dependencies(filename.parent / line[8:])
            else:
                break
        return d
            
def yaml_dependencies(filenames):
    deps = set()
    for f in filenames:
        deps |= yaml_load_dependencies(Path(f))
    
    return deps
