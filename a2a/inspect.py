import builtins

from a2a.utils.globals import main_globals

for x in main_globals:
    builtins.__dict__[x] = main_globals[x]
