import os
from importlib.machinery import SourceFileLoader

def add_args(args1, args2):
    for k, v in args2.__dict__.items():
        args1.__dict__[k] = v

def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    return SourceFileLoader(module_name, filename).load_module()
