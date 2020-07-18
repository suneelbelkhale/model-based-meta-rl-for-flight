import torch
import torch.optim as optim
import torch.distributions as D
import torch.utils.data as data
import torch.distributions.constraints as C
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List

from mbrl.utils.torch_utils import activation_map, layer_map, reshape_map

all_layer_map = {**layer_map, **activation_map, **reshape_map}


# default layer
class LayerParams:
    def __init__(self, ltype, **kwargs):
        self.type = ltype if type(ltype) == type else all_layer_map[ltype.lower()]
        self.kwargs = kwargs

    def to_module_list(self, **opt):
        return self.type(**self.kwargs)


class SequentialParams(LayerParams):
    def __init__(self, layer_params: List[LayerParams]):
        self.params = layer_params
        self.length = len(layer_params)

    def to_module_list(self, as_sequential=True, **opt):
        block = []
        for i in range(self.length):
            block.append(self.params[i].to_module_list())

        if as_sequential:
            return nn.Sequential(*block)
        else:
            return block