import numpy as np
import torch

from mbrl.utils import abstract
from mbrl.utils.python_utils import AttrDict
from mbrl.utils.torch_utils import to_torch, torch_clip


class EnvSpec(abstract.BaseClass):

    def __init__(self, params):
        names_shapes_limits_dtypes = list(params.names_shapes_limits_dtypes)
        names_shapes_limits_dtypes += [('done', (), (False, True), np.bool)]

        self._names_to_shapes = AttrDict()
        self._names_to_limits = AttrDict()
        self._names_to_dtypes = AttrDict()
        for name, shape, limit, dtype in names_shapes_limits_dtypes:
            self._names_to_shapes[name] = shape
            self._names_to_limits[name] = limit
            self._names_to_dtypes[name] = dtype

    @property
    @abstract.abstractmethod
    def observation_names(self):
        """
        Returns:
            list(str)
        """
        raise NotImplementedError

    @property
    def output_observation_names(self):
        return self.observation_names

    @property
    def goal_names(self):
        """
        The only difference between a goal and an observation is that goals are user-specified

        Returns:
            list(str)
        """
        return tuple()

    @property
    @abstract.abstractmethod
    def action_names(self):
        """
        Returns:
            list(str)
        """
        raise NotImplementedError

    @property
    def names(self):
        """
        Returns:
            list(str)
        """
        return self.observation_names + self.goal_names + self.action_names

    @property
    def names_to_shapes(self):
        """
        Knowing the dimensions is useful for building neural networks

        Returns:
            AttrDict
        """
        return self._names_to_shapes

    @property
    def names_to_limits(self):
        """
        Knowing the limits is useful for normalizing data

        Returns:
            AttrDict
        """
        return self._names_to_limits

    @property
    def names_to_dtypes(self):
        """
        Knowing the data type is useful for building neural networks and datasets

        Returns:
            AttrDict
        """
        return self._names_to_dtypes

    def map_to_types(self, input_dict: AttrDict):
        def map_helper(key, arr_in):
            if isinstance(arr_in, np.ndarray):
                return arr_in.astype(self._names_to_dtypes[key])

        input_dict = input_dict.copy()
        input_dict.leaf_kv_modify(map_helper)
        return input_dict

    def assert_has_names(self, names):
        for n in names:
            assert n in self.names, "%s not in env spec names" % n

    def limits(self, names):
        lower, upper = [], []
        for name in names:
            shape = self.names_to_shapes[name]
            typ = self.names_to_dtypes[name]
            assert len(shape) >= 1
            l, u = self.names_to_limits[name]
            lower += [np.broadcast_to(l, shape).astype(typ)]
            upper += [np.broadcast_to(u, shape).astype(typ)]
        return np.array(lower), np.array(upper)

    def clip(self, d, names):
        low, high = self.limits(names)
        for i in range(len(names)):
            name = names[i]
            if isinstance(d[name], torch.Tensor):
                l = to_torch(low[i], device=d[name].device)
                h = to_torch(high[i], device=d[name].device)
                d[name] = torch_clip(d[name], l, h)
            else:
                d[name] = np.clip(d[name], low, high)

    def dims(self, names):
        return np.array([np.prod(self.names_to_shapes[name]) for name in names])

    def dim(self, names):
        return np.sum(self.dims(names))

    def get_uniform(self, names, batch_size, torch_device=None):
        low, upp = self.limits(names)

        d = AttrDict()
        for i, name in enumerate(names):
            d[name] = np.random.uniform(low[i], upp[i], size=[batch_size] + list(low[i].shape))
            d[name] = d[name].astype(low[i].dtype)

        if torch_device is not None:
            d.leaf_modify(lambda x: torch.from_numpy(x).to(torch_device))

        return d

    def get_midpoint(self, names, batch_size, torch_device=None):
        low, upp = self.limits(names)

        d = AttrDict()
        for i, name in enumerate(names):
            d[name] = ((low[i] + upp[i]) / 2)[None].repeat(batch_size, axis=0)
            d[name] = d[name].astype(low[i].dtype)

        if torch_device is not None:
            d.leaf_modify(lambda x: torch.from_numpy(x).to(torch_device))

        return d

    def get_zeros(self, names, batch_size, torch_device=None):

        d = AttrDict()
        for name in names:
            d[name] = np.zeros([batch_size] + list(self.names_to_shapes[name]))
            d[name] = d[name].astype(self.names_to_dtypes[name])

        if torch_device is not None:
            d.leaf_modify(lambda x: torch.from_numpy(x).to(torch_device))

        return d