import numpy as np

from mbrl.envs.env_spec import EnvSpec
from mbrl.utils import abstract


class LatentEnvSpec(EnvSpec):
    @property
    @abstract.overrides
    def observation_names(self):
        """
        Returns:
            list(str)
        """
        return ['obs', 'prev_obs', 'prev_act', 'latent']

    @property
    def output_observation_names(self):
        return ['next_obs', 'next_obs_sigma']

    @property
    def goal_names(self):
        """
        The only difference between a goal and an observation is that goals are user-specified

        Returns:
            list(str)
        """
        return ['goal_obs']

    @property
    @abstract.overrides
    def action_names(self):
        """
        Returns:
            list(str)
        """
        return ['act']
