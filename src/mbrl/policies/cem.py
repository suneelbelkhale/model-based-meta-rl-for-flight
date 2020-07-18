import torch

from mbrl.policies.policy import Policy
from mbrl.policies.random_shooting import OptimizerPolicy
from mbrl.utils import abstract

# should not be used in traditional training loop
from mbrl.utils.python_utils import AttrDict
from mbrl.utils.torch_utils import split_dim


class CEM(OptimizerPolicy):

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        self._pop_size = params.popsize  # total candidates to resample
        self._horizon = params.horizon  # environment horizon
        self._act_dim = params.act_dim  # environment horizon
        self._max_iters = params.max_iters  # max iterations of optimization / resampling
        self._num_elites = params.num_elites  # top candidate to target next distribution
        self._epsilon = params.epsilon  # minimum allowed variance
        self._alpha = params.alpha  # momentum per iter

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

        def resample_and_flatten(vs):
            old_acseq = vs[0]
            mean, std = vs[1], vs[2]
            sample = torch.randn_like(old_acseq) * std + mean
            d = AttrDict(act=sample)
            self._env_spec.clip(d, ['act'])
            return d.act.view([-1] + list(old_acseq.shape[2:]))

        best_initial_act = None
        results = None
        for i in range(self._max_iters):
            # run the model
            results = model(action_sequence, observation, goal)

            # view as (B, Pop, ...)
            action_sequence.leaf_modify(lambda x: split_dim(x, 0, [batch_size, self._pop_size]))
            results.leaf_modify(lambda x: split_dim(x, 0, [batch_size, self._pop_size]))

            results.order = torch.argsort(results.costs, dim=1)  # lowest to highest

            best = results.order[:, :self._num_elites]
            best = best.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, self._horizon, self._act_dim))
            best_act_seq = action_sequence.leaf_apply(lambda x: torch.gather(x, 1, best))
            best_initial_act = best_act_seq.leaf_apply(lambda x: x[:, 0, 0])  # where x is (B, Pop, H ...)
            means = best_act_seq.leaf_apply(lambda x: x.mean(1, keepdim=True))
            stds = best_act_seq.leaf_apply(lambda x: x.std(1, keepdim=True))

            if i < self._max_iters - 1:
                # resampling
                action_sequence = AttrDict.leaf_combine_and_apply([action_sequence, means, stds], resample_and_flatten)

        # act is (B, actdim)
        best_initial_act.action_sequence = action_sequence  # (B*Pop, horizon, actdim)
        best_initial_act.results = results

        return best_initial_act
