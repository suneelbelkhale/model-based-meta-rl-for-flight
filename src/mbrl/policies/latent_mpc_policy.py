from typing import Callable

import numpy as np
import torch

from mbrl.envs.env_spec import EnvSpec
from mbrl.policies.policy import Policy
from mbrl.policies.random_shooting import OptimizerPolicy
from mbrl.utils import abstract
from mbrl.utils.mpc_utils import rollout
from mbrl.utils.python_utils import AttrDict
from mbrl.utils.torch_utils import to_torch


class LatentMPCPolicy(Policy):

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        # self._infer_latent = params.infer_latent
        self._num_particles = params.num_particles
        # self._is_probabilistic = params.is_probabilistic
        # self._num_nets = params.num_nets
        self._horizon = params.horizon

        optim_cls = params.optimizer_cls
        optim_params = params.optimizer_params
        self._optimizer_policy = optim_cls(optim_params, self._env_spec)
        assert isinstance(self._optimizer_policy, OptimizerPolicy)

        self._set_fn("_cost_fn", params.cost_function,
                     Callable[[AttrDict, AttrDict, AttrDict], torch.Tensor])  # (obs_seq, goal_seq) -> cost
        self._set_fn("_advance_obs_fn", params.advance_obs_function,
                     Callable[[AttrDict, AttrDict, EnvSpec], AttrDict])  # (obs, model_out) -> next_obs


    @abstract.overrides
    def _init_setup(self):
        pass

    @abstract.overrides
    def warm_start(self, model, observation, goal):
        pass

    @abstract.overrides
    def get_action(self, model, observation, goals, batch=True):
        """
        Args:
            model (Model):
            observation (AttrDict):
            goal (AttrDict):
            batch (bool):

        Returns:
            AttrDict
        """
        observation = observation.leaf_apply(lambda tin: to_torch(tin, device=model.device, check=True))
        goals = goals.leaf_apply(lambda tin: to_torch(tin, device=model.device, check=True))
        def model_call(acs, obs, gls): return self.eval_act_sequence(model, acs, obs, gls)

        # will contain at least {action_sequence, results {trajectory, costs, order}}
        model.eval()
        cem_outputs = self._optimizer_policy.get_action(model_call, observation, goals, batch=batch)
        return cem_outputs

    def eval_act_sequence(self, model, action_seq, observations, goals):
        """ Finds predicted trajectory for a given batch of ac_sequences on given initial obs and prev_obs vectors
        Arguments:
            model: the underlying dynamics model
            observations: dotmap:(N x), initial observations (state, state hist, act hist, latent hist)
            action_seq: (N x H x dotmap{}), action sequences per initial observation
            goals: should be shape (N, H+1, dO) or broadcastable
        Returns: dictionary{ctrl_seq, traj_seq, cost, }
        """

        # TODO implement multiple particles

        # run the model forward on obs_start
        all_obs, all_mouts = rollout(self._env_spec, model, observations, action_seq, self._advance_obs_fn)

        # first unsqueezes and then concats
        all_obs = AttrDict.leaf_combine_and_apply(all_obs,
                                                  func=lambda vs: torch.cat(vs, dim=1),
                                                  map_func=lambda arr: arr.unsqueeze(1))
        all_mouts = AttrDict.leaf_combine_and_apply(all_mouts,
                                                  func=lambda vs: torch.cat(vs, dim=1),
                                                  map_func=lambda arr: arr.unsqueeze(1))
        costs = self._cost_fn(all_obs, goals, action_seq, all_mouts)

        return AttrDict(
            trajectory=all_obs,
            costs=costs  # (N,)
        )

