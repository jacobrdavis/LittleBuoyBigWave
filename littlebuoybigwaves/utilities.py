"""
Utilities module for LittleBuoyBigWaves.
"""

#TODO:
# - need to implement default namespaces

__all__ = [
    "get_config",
]

import types

import toml


def get_config():
    with open('config.toml', 'r') as f:
        config = toml.load(f)

    return config


def get_var_namespace(subset=None):
    try:  #TODO: might be better soln here
        if subset is None:
            config = get_config()['littlebuoybigwaves']
        else:
            config = get_config()['littlebuoybigwaves'][subset]
        var_namespace = types.SimpleNamespace(**config['vars'])

    except FileNotFoundError as error:
        print(f'{error}.  Using default SimpleNamespace for {subset}.')
        #TODO: DEFAULT HERE!
        var_namespace = None
    return var_namespace
