from typing import Callable

import numpy as np
import torch

from mbrl.experiments import logger
from mbrl.policies.policy import Policy
from mbrl.utils import abstract
from mbrl.utils.mpc_utils import rollout
from mbrl.utils.python_utils import AttrDict
from mbrl.utils.torch_utils import to_torch

import mbrl.utils.ros_utils as ru
from mbrl.utils.ros_utils import RosTopic

# policy that feeds actions from a ros topic
class RosPolicy(Policy):

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        self._ros_action_topic = params.ros_action_topic
        self._ros_action_type = params.ros_action_type
        self._msg_to_numpy_fn = params.msg_to_numpy_fn

        self._background_policy_cls = params.get("background_policy_cls", None)
        self._background_policy_params = params.get("background_policy_params", None)

    @abstract.overrides
    def _init_setup(self):
        self._bg_policy = None
        if self._background_policy_cls is not None and self._background_policy_params is not None:
            self._bg_policy = self._background_policy_cls(self._background_policy_params, self._env_spec)
            assert isinstance(self._bg_policy, Policy)
            logger.debug("[RosPolicy] Using background policy class: %s" % self._background_policy_cls)

        self._latest_action = None
        self._ros_motion_sub = ru.bind_subscriber(RosTopic(self._ros_action_topic, self._ros_action_type),
                                                  self.action_callback)

        logger.debug("[RosPolicy] Waiting for action messages to translate.")
        while self._latest_action is None:
            pass


    @abstract.overrides
    def warm_start(self, model, observation, goal):
        pass

    def action_callback(self, msg):
        self._latest_action = self._msg_to_numpy_fn(msg)
        assert isinstance(self._latest_action, np.ndarray), "[RosPolicy] msg_to_numpy must return an np array"

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
        if self._bg_policy is not None:
            action_dict = self._bg_policy.get_action(model, observation, goals, batch=batch)
        else:
            action_dict = AttrDict()

        if batch:
            act = np.tile(self._latest_action[None], (observation.obs.shape[0], 1))
        else:
            act = self._latest_action

        # save a copy of the bg action if its there
        if "act" in action_dict.keys():
            action_dict.bg_act = action_dict.act

        action_dict.act = to_torch(act)

        return action_dict
