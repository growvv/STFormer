# Copyright (c) CAIRI AI Lab. All rights reserved


from .simvp import SimVP

method_maps = {
    'simvp': SimVP,
}

__all__ = [
    'method_maps', 'SimVP',
]