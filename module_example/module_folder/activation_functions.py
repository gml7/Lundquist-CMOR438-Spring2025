import numpy as np


__all__ = [
    'sigmoid', 
    'linear',
    'sign',
#    'no_func' # `__init__.py` will try to import this but raise an AttributeError since it isn't defined in this file
]

def sigmoid():
    return 1

def linear(x):
    return x

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1
    
# Since it's not named in `__all__`, you can't call this from another file even after importing module_folder
def test_func():
    return "test"