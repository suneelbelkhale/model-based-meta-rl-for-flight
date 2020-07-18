"""
This is where all the neural network magic happens!

Whether this model is a Q-function, policy, or dynamics model, it's up to you.
"""
from typing import Callable

from dotmap import DotMap
import torch
from mbrl.experiments import logger
from mbrl.utils import abstract
from mbrl.utils.python_utils import AttrDict


class Model(torch.nn.Module):

    def __init__(self, params, env_spec, dataset_train):
        super(Model, self).__init__()

        self._env_spec = env_spec
        self._dataset_train = dataset_train

        if "cuda" not in str(params.device) or torch.cuda.is_available():
            self._device = torch.device(params.device)
        else:
            self._device = torch.device("cpu")

        # TODO logger
        logger.debug("Model using device: {}".format(self._device))

        keys = params.leaf_keys()
        # first thing called in model forward (inputs) -> new_inputs
        if 'preproc_fn' in keys:
            self.set_fn("_preproc_fn", params.preproc_fn, Callable[[AttrDict], AttrDict])  # mutation okay

        # updates model outputs (inputs, model_outputs) -> new_model_outputs
        if 'postproc_fn' in keys:
            self.set_fn("_postproc_fn", params.postproc_fn, Callable[[AttrDict, AttrDict], AttrDict])  # mutation okay

        # gets model loss (inputs, outputs, model_outputs) -> loss torch tensor
        if 'loss_fn' in keys:
            self.set_fn("_loss_fn", params.loss_fn, Callable[[AttrDict, AttrDict, AttrDict], torch.Tensor])

        self._init_params_to_attrs(params)
        self._init_setup()

    # proper function typing
    def set_fn(self, name, func, ftype):
        if func is None:
            return

        def _internal_setter(fn: ftype):
            self.__setattr__(name, fn)
        _internal_setter(func)

    def has_fn(self, name):
        return hasattr(self, name) and self.__getattr__(name) is Callable

    @property
    def device(self):
        return self._device

    @property
    def env_spec(self):
        return self._env_spec

    @abstract.abstractmethod
    def _init_params_to_attrs(self, params):
        pass

    @abstract.abstractmethod
    def _init_setup(self):
        pass

    @abstract.abstractmethod
    def warm_start(self, model, observation, goal):
        raise NotImplementedError

    def forward(self, inputs, obs_lowd=None, training=False):

        """
        Args:
            inputs (DotMap):
            training (bool):

        Returns:
            outputs (DotMap):
        """
        pass

    def restore_from_checkpoint(self, checkpoint):
        self.load_state_dict(checkpoint['model'])

    def restore_from_file(self, fname):
        self.restore_from_checkpoint(torch.load(fname))
