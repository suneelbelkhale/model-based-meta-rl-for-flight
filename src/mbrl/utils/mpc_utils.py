from itertools import tee

import numpy as np
import torch

from mbrl.envs.env_spec import EnvSpec
from mbrl.experiments import logger
from mbrl.utils.python_utils import AttrDict as d
from mbrl.utils.torch_utils import advance_history


# template advance obs function, used for LATENT experiments
# returns the next OBSERVATION, not to be confused with inputs
# make sure to clip the observations here
def latent_advance_obs_fn(inputs: d, model_outputs: d, env_spec: EnvSpec) -> d:
    env_spec.clip(model_outputs, ["next_obs"])
    return d(
        obs=model_outputs.next_obs.mean(dim=-2),  # mean over all models
        prev_obs=advance_history(inputs.prev_obs, inputs.obs),
        prev_act=advance_history(inputs.prev_act, inputs.act),
        latent=inputs.latent,
    )


# template mpc cost function. used for sampling action sequences
def default_mpc_cost_fn(obs_seq: d, goal_seq: d, action_seq: d, model_out_seq: d) -> torch.Tensor:
    assert obs_seq.obs.shape == goal_seq.goal_obs.shape  # (N, H+1, obsdim)
    # this std is some scaled version of the observation standard deviation
    std = model_out_seq.next_obs_sigma[:, :1, 0]  # (N, 1, obsdim) TODO non deterministic
    normalized = torch.abs(obs_seq.obs - goal_seq.goal_obs) / std
    return normalized.sum(-2).mean(-1)  # (N,)


# used for online inference
def latent_obs_to_output_obs_fn(obs: d) -> d:
    return d(next_obs=obs.obs, next_obs_sigma=1e-20 * torch.ones_like(obs.obs))


def rollout(env_spec, model, start_obs, action_seq, advance_obs_fn):
    curr_obs = start_obs
    all_obs = [start_obs]
    all_mouts = []

    for i in range(action_seq.act.shape[1]):
        inputs = d()
        for name in env_spec.observation_names:
            inputs[name] = curr_obs[name]

        # TODO allow for only a subset of actions to be optimized
        for name in env_spec.action_names:
            inputs[name] = action_seq[name][:, i]

        model_outputs = model(inputs)
        next_obs = advance_obs_fn(inputs, model_outputs, env_spec)  # should do any clipping
        all_obs.append(next_obs)
        all_mouts.append(model_outputs)
        curr_obs = next_obs

    return all_obs, all_mouts


######## trajectory stuff ########
class Trajectory:
    # type can be absolute, relative, or velocity
    # metric represents evaluation
    def __init__(self, traj_type, iterator, metric, is_waiting_func, origin, loop=False):
        self.type = traj_type
        _, self.original, self.iterator = tee(iterator, 3)
        self.metric = metric
        self.is_waiting = is_waiting_func
        self.curr_goal = None
        self.origin = origin
        self.loop = loop
        self.completed = False  # if we have moved through the whole trajectory
        self.i = 0

        if traj_type == 'absolute':
            pass
        elif traj_type == 'relative':
            pass
        elif traj_type == 'velocity':
            pass

    def next(self, state):
        # ensures that goal is set after at least one call to next (used by pickup code)
        self.metric(state, state, self.i)

        # metric first
        if self.curr_goal is None or self.metric(state, self.curr_goal, self.i):
            # in this case, we have reached the current goal
            try:
                self.curr_goal = next(self.iterator)
                self.i += 1  # represents the number of distinct targets we have seen
            except StopIteration as e:
                if self.loop:
                    self.reset()
                    try:
                        self.curr_goal = next(self.iterator)
                    except StopIteration as e:
                        # in this case we cannot loop around so just give up
                        self.completed = True
                else:
                    self.completed = True

        # goals are always relative to [0,0,0], not true origin
        # states are always relative to true origin
        # we correct for that here
        return self.curr_goal + self.origin

    # rolls out goals for steps without affecting the original
    def try_next_n(self, num):
        _, self.iterator, try_iter = tee(self.iterator, 3)

        goals = []
        try:
            for j in range(num):
                # stop if we ever reach a waiting point (corresponds to the point before)
                if self.is_waiting(self.i + j):
                    raise StopIteration
                next_goal = next(try_iter)
                goals.append(next_goal)
        except StopIteration:
            # logger.debug("stopped at %d elements" % len(goals))
            if len(goals) == 0 and self.curr_goal is not None:
                goals = [self.curr_goal] * num
            elif len(goals) > 0:
                last_filled = len(goals) - 1
                for i in range(num - len(goals)):
                    goals.append(goals[last_filled])  # fill the remainder
            else:
                raise NotImplementedError("[Traj]: try next n not defined when curr_goal is None")

        return np.stack(goals)

    def get_i(self):
        return self.i

    def is_finished(self):
        return self.completed

    def reset(self, zero_at=None, new_iterator=None):
        if zero_at is not None:
            self.origin = zero_at

        if new_iterator is not None:
            self.iterator = new_iterator
        else:
            _, self.original, self.iterator = tee(self.original, 3)

        self.curr_goal = None
        self.completed = False
        self.i = 0

        logger.debug("RESET AT: " + str(self.origin[:2]))


class Metrics:
    @staticmethod
    def distance_thresh(thresh, mask=None):

        def func(state, goal, i):
            nonlocal mask
            if mask is None:
                mask = np.ones(state.shape)
            return np.linalg.norm(np.multiply(state, mask) - goal)

        return Metrics.ext_function_thresh(thresh, func)

    @staticmethod
    def individual_distance_thresh(thresh_array, mask=None):

        def func(state, goal, i):
            nonlocal mask
            if mask is None:
                mask = np.ones(state.shape)
            abs_diff = np.abs(np.multiply(state, mask) - goal)
            logger.debug("[TEMP] offsets / within: " + str(abs_diff) + " / " + str(abs_diff <= thresh_array))
            return np.all(abs_diff <= thresh_array)

        return func

    @staticmethod
    def sequential():
        return lambda state, goal, i: True

    @staticmethod
    def wait_start_sequential(initial_func):
        # runs sequentially after the first
        return lambda state, goal, i: initial_func(state, goal, i) if i <= 1 else True

    @staticmethod
    def ext_function_thresh(thresh, func):

        def metric(state, goal, i):
            assert state.shape == goal.shape
            dist = func(state, goal)
            # termination condition
            return dist < thresh

        return metric


############ SOME SAMPLE TRAJECTORIES #############

class SampleTrajectories:

    @staticmethod
    def delta_trajectory(start_corner, *deltas):
        waypoints = [np.array(list(start_corner))]

        curr = waypoints[-1]
        for i in range(len(deltas)):
            curr = curr + np.array(list(deltas[i]))
            waypoints.append(curr)

        return iter(waypoints)

    @staticmethod
    def even_interpolate(waypoints, N):
        waypoints = list(waypoints)

        new_points = [waypoints[0]]

        for i in range(len(waypoints) - 1):
            wi = waypoints[i]
            wip1 = waypoints[i + 1]
            step = (wip1 - wi) / N
            new_points.extend([wi + j * step for j in range(1, N + 1)])

        return iter(new_points)

    @staticmethod
    def circle_trajectory(center, ax1, ax2, radius, num_points_per_rot=100, num_rot=1):
        assert ax1 != ax2 and max(ax1, ax2) < len(center)

        theta_inc = 2. * np.pi / num_points_per_rot

        waypoints = []

        for i in range(num_points_per_rot * num_rot):
            theta = i * theta_inc

            nextpt = list(center)
            nextpt[ax1] = center[ax1] + radius * np.cos(theta)
            nextpt[ax2] = center[ax2] + radius * np.sin(theta)

            waypoints.append(np.array(nextpt))

        return iter(waypoints)

    @staticmethod
    def sinusoid_trajectory(center, amp, ax, num_points_per_rot=100, num_rot=1):
        assert ax < len(center)

        theta_inc = 2. * np.pi / num_points_per_rot

        waypoints = []

        for i in range(num_points_per_rot * num_rot):
            theta = i * theta_inc

            nextpt = list(center)
            nextpt[ax] = center[ax] + amp * np.sin(theta)

            waypoints.append(np.array(nextpt))

        return iter(waypoints)

    @staticmethod
    def slither_trajectory(center, amp, ax, ax_vert, vert_amp, num_points_per_rot=100, num_rot=1):
        assert ax < len(center)
        assert ax_vert < len(center)

        theta_inc = 2. * np.pi / num_points_per_rot

        waypoints = []

        for i in range(num_points_per_rot * num_rot):
            theta = i * theta_inc

            nextpt = list(center)
            nextpt[ax] = center[ax] + amp * np.sin(theta)
            nextpt[ax_vert] = center[ax_vert] + vert_amp - i * 2 * vert_amp / (
                    num_points_per_rot * num_rot)  # progress linearly in this direction

            waypoints.append(np.array(nextpt))

        return iter(waypoints)

    @staticmethod
    def figure8_trajectory(start_x, start_y, start_side, x_amp, z_factor=0.2, num_points_per_rot=100, num_rot=1):
        theta_inc = 2. * np.pi / num_points_per_rot

        waypoints = []

        cZ0 = 1. / start_side

        in_side = start_side / (1 - z_factor)
        out_side = start_side / (1 + z_factor)

        amp_in = in_side - start_side
        amp_out = start_side - out_side

        for i in range(num_points_per_rot * num_rot):
            theta_side = i * theta_inc
            theta_x = 2 * i * theta_inc

            nextpt = [start_x, start_y, start_side]
            nextpt[2] += amp_in * np.sin(theta_side) if np.sin(theta_side) >= 0 else amp_out * np.sin(theta_side)
            nextpt[0] += x_amp * np.sin(theta_x)

            nextpt[2] = nextpt[2] ** 2  # square to get area

            waypoints.append(np.array(nextpt))

        return iter(waypoints)
