import torch

from mbrl.policies.policy import Policy
from mbrl.utils import abstract

# should not be used in traditional training loop
from mbrl.utils.python_utils import AttrDict
from mbrl.utils.torch_utils import split_dim


class OptimizerPolicy(Policy):
    pass


class RandomShooting(OptimizerPolicy):

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        self._pop_size = params.popsize
        self._horizon = params.horizon
        self._act_dim = params.act_dim
        self._action_names = params.action_names if "action_names" in params.keys() else self._env_spec.action_names
        self._action_names_unoptimized = [a for a in self._env_spec.action_names if a not in self._action_names]
        self._num_actions = len(self._action_names)

    @abstract.overrides
    def _init_setup(self):
        pass

    @abstract.overrides
    def warm_start(self, model, observation, goal):
        pass

    @abstract.overrides
    def get_action(self, model, observation, goal, batch=False):
        """Optimizes the cost function provided in setup().
        Arguments:
            model: must be callable(action_sequence, observation, goal) and return cost (torch array)
                    where action is at AttrDict consisting of keys only in self.action_names
            observation: {}
            goal: {goal_obs} where goal_obs must be N x H+1 x ...
            batch:

        Returns:
            AttrDict with {action_sequence, results {costs, order} }
        """
        # generate random sequence
        batch_size = goal.goal_obs.shape[0]  # requires goal_obs to be a key
        device = goal.goal_obs.device

        if not batch:
            observation = observation.leaf_apply(lambda arr: arr.unsqueeze(0).repeat_interleave(self._pop_size, dim=0))
            goal = goal.leaf_apply(lambda arr: arr.unsqueeze(0).repeat_interleave(self._pop_size, dim=0))
        else:
            observation = observation.leaf_apply(lambda arr: arr.repeat_interleave(self._pop_size, dim=0))
            goal = goal.leaf_apply(lambda arr: arr.repeat_interleave(self._pop_size, dim=0))

        action_sequence = self._env_spec.get_uniform(self._action_names,
                                                     batch_size=batch_size * self._pop_size * self._horizon)
        action_sequence.leaf_modify(lambda x: split_dim(torch.from_numpy(x).to(device), dim=0,
                                                        new_shape=[batch_size * self._pop_size, self._horizon]))

        # run the model
        results = model(action_sequence, observation, goal)

        # view as (B, Pop, ...)
        action_sequence.leaf_modify(lambda x: split_dim(x, 0, [batch_size, self._pop_size]))
        results.leaf_modify(lambda x: split_dim(x, 0, [batch_size, self._pop_size]))

        results['order'] = torch.argsort(results.costs, dim=1)  # lowest to highest (best to worst)
        best = results.order[:, :1].unsqueeze(-1).unsqueeze(-1).expand((-1, -1, self._horizon, self._act_dim))
        best_act_seq = action_sequence.leaf_apply(lambda x: torch.gather(x, 1, best))
        best_initial_act = best_act_seq.leaf_apply(lambda x: x[:, 0, 0])  # where x is (B, Pop, H ...)

        return AttrDict(
            act=best_initial_act,
            action_sequence=action_sequence,
            results=results
        )
