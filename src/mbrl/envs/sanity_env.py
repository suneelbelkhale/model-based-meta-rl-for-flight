from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import rospy

from mbrl.envs.env import Env
from mbrl.experiments import logger
from mbrl.utils import abstract
from mbrl.utils import ros_utils as ru

import os

from crazyflie.msg import CFMotion
from crazyflie.sim.point_mass_cf_simulator import PointMassCrazyflieSimulator
from gym import spaces

import graphics as gr

import random
import numpy as np

from mbrl.utils.np_utils import advance_history_np
from mbrl.utils.python_utils import AttrDict
from mbrl.utils.torch_utils import to_numpy

sanity = [1/0.31336035, 1/0.31142008, 1/0.15706026]
WEIGHTS = sanity

INITIAL_GOAL_POS = [0.5, 0.5, 0.05]  # move to center pixel
PIX_TOL = [0.005, 0.005, 0.0025]  # move to center pixel

WIDTH = 640
HEIGHT = 480


def move_to(obj, old_pt, pt):
    dx = pt.getX() - old_pt.getX()
    dy = pt.getY() - old_pt.getY()
    obj.move(dx, dy)


def point_from(arr):
    x = int(arr[0] * WIDTH)
    y = int(arr[1] * HEIGHT)
    return gr.Point(x, y)


class CFSanityEnv(Env):
    def __init__(self, params, env_spec):
        super().__init__(params, env_spec)
        # Setup the state indices

        # Setup the parameters
        self._dt = params.dt
        self._step_dt = params.step_dt
        self._ros_prefix = params.ros_prefix
        self._lag = params.lag
        self._use_random_goal = params.use_random_goal
        self._obs_hist_len = params.obs_hist_len
        self._act_hist_len = params.act_hist_len
        self.horizon = params.horizon
        self._num_latent = params.num_latent

        ru.setup_ros_node('MBRLNode', anonymous=True)
        self._sim = PointMassCrazyflieSimulator(self._ros_prefix, self._dt, use_ros=False, lag=self._lag,
                                                with_figure=False, num_latent=self._num_latent)
        self.ob_lb = np.array([self._sim.X_RANGE[0], self._sim.Y_RANGE[0], self._sim.Z_RANGE[0]])
        self.ob_ub = np.array([self._sim.X_RANGE[1], self._sim.Y_RANGE[1], self._sim.Z_RANGE[1]])

        self.observation_space = spaces.Box(self.ob_lb, self.ob_ub)

        self.x_dim = 3  # pixel x,y,area
        self.u_dim = 3  # vx, vy, vz

        self.ac_goal_pos = np.zeros((self.x_dim,))
        self.np_weights = np.array(WEIGHTS)
        self.cost_tol = np.array(PIX_TOL).dot(self.np_weights) # (weighted cost)

        self._obs = np.zeros((self.x_dim,))
        self._prev_obs = None
        self._prev_act = None

        self._latent_idx = -1

        self.cost = 0
        self.done = False

        # ros callbacks
        self.ac_lb = np.array([-1., -1., -0.5])
        self.ac_ub = np.array([1., 1., 0.5])

        self.action_space = spaces.Box(self.ac_lb, self.ac_ub)

        self._step_rate = rospy.Rate(1. / self._step_dt)

        # graphics
        self._win = gr.GraphWin("Simulator", WIDTH, HEIGHT)
        self._g_obs = gr.Circle(gr.Point(0, 0), 2)
        self._g_obs.draw(self._win)
        self._g_targ = gr.Circle(gr.Point(0, 0), 2)
        self._g_targ.draw(self._win)

        self._g_lines = []
        self._curr_trajectory = []

        self.reset()

    def render_update(self, action):
        pt = point_from(self._obs)
        move_to(self._g_obs, self._g_obs.getCenter(), pt)

        goal_pt = point_from(self.ac_goal_pos)
        move_to(self._g_targ, self._g_targ.getCenter(), goal_pt)

        if len(self._g_lines) > 0:
            start_pt = self._g_lines[-1].getP2()
        else:
            start_pt = pt

        self._g_lines.append(gr.Line(start_pt, pt))
        self._g_lines[-1].setFill("red")
        self._g_lines[-1].setWidth(1)
        self._g_lines[-1].draw(self._win)

        # predicted trajectory rendering
        if 'results/trajectory/obs' in action.leaf_keys():
            # clear
            for line in self._curr_trajectory:
                line.undraw()
            self._curr_trajectory = []

            # refill
            idx = action.results.order[0, 0]
            trajectory = action.results.trajectory.obs[0, idx]
            for i in range(trajectory.shape[0] - 1):
                line = gr.Line(point_from(trajectory[i]), point_from(trajectory[i+1]))
                line.setFill("green"); line.setWidth(1); line.draw(self._win)
                self._curr_trajectory.append(line)

    def render_clear(self):
        for line in self._g_lines:
            line.undraw()
        self._g_lines = []

        for line in self._curr_trajectory:
            line.undraw()
        self._curr_trajectory = []

        self._g_obs.undraw()
        self._g_obs = gr.Circle(gr.Point(0, 0), 2)
        self._g_obs.setFill("red")
        self._g_obs.draw(self._win)

        self._g_targ.undraw()
        self._g_targ = gr.Circle(gr.Point(0, 0), 2)
        self._g_targ.setFill("blue")
        self._g_targ.draw(self._win)

    def _step(self, u, no_rew_pub=False, step_extra_func=None, buffer_window_size=-1, **kwargs):

        self._enforce_dimension(self._obs, u)

        # Limit the control inputs
        u0 = np.clip(u, self.ac_lb, self.ac_ub)

        last_state = np.copy(self._obs)

        # update vx, vy, dz (does NOT publish)
        self._set_motion(u0[0], u0[1], u0[2])
        vec = self._sim.step(self._curr_motion)

        self._prev_obs = advance_history_np(self._prev_obs[None], self._obs[None])[0]
        self._prev_act = advance_history_np(self._prev_act[None], u0[None])[0]

        self._obs[0] = vec.vector.x
        self._obs[1] = vec.vector.y
        self._obs[2] = vec.vector.z

        self.cost = self.get_cost()
        self.done = self.cost <= self.cost_tol

        if np.any(np.isnan(self._obs)):
            logger.debug('CF SANITY: NAN POSITION')

        logger.debug("[SanityEnv]: lat=%d, AC: %s, OB: %s, NEXT_OB: %s, GOAL: %s, REW: %f" % (
            self._latent_idx,
            np.array2string(u0, separator=', '),
            np.array2string(last_state, separator=', '),
            np.array2string(self._obs, separator=', '),
            np.array2string(self.ac_goal_pos, separator=', '),
            -self.cost)
        )

        ## EXTRA FUNCTIONALITY for inferring latent online
        ret = None
        if step_extra_func is not None:
            def exit_condition(iters): return self._step_rate.remaining().to_sec() < 1e-7 \
                                           and self._step_rate.sleep() is None
            ret = step_extra_func(exit_cond=exit_condition,
                                  buffer_window_size=buffer_window_size)  # returns at or before sleep_end_time
            mu = ret[0][0]
            logsig = ret[1][0]  # TODO do something with these
        else:
            self._step_rate.sleep()  # sleep until next iter

        return self.get_obs(), self.get_goal(), self.done

    def _set_motion(self, vx, vy, vz):
        motion = CFMotion()
        motion.x = vx
        motion.y = vy
        motion.yaw = 0 # yaw disabled for now
        motion.dz = vz
        motion.is_flow_motion = True

        self._curr_motion = motion

    def _set_target(self, xf):
        self.ac_goal_pos = xf

    def _set_target_relative(self, xf):
        self.ac_goal_pos = xf + self._obs

    def _enforce_dimension(self, x, u):
        assert len(x) == self.x_dim and len(u) == self.u_dim, "x: %d, %d .. u: %d, %d" % (len(x), self.x_dim, len(u), self.u_dim)

    @abstract.overrides
    def step(self, action, **kwargs):
        action = action.leaf_apply(lambda tin: to_numpy(tin, check=True))
        self.render_update(action)
        return self._step(action.act[0], **kwargs)

    @abstract.overrides
    def reset(self, ret_latent=False):
        self._curr_motion = CFMotion()
        self._curr_motion.is_flow_motion = True

        vec, latent_idx = self._sim.reset([random.random(), random.random(), 0.5 * random.random()], ret_latent=True)
        self._latent_idx = latent_idx

        self._obs[0] = vec.vector.x
        self._obs[1] = vec.vector.y
        self._obs[2] = vec.vector.z

        self.render_clear()

        obs, goal = self.reset_model()
        if ret_latent:
            return obs, goal, latent_idx
        else:
            return obs, goal

    def reset_model(self):
        if self._use_random_goal:
            self.ac_goal_pos = np.random.uniform(self.ob_lb, self.ob_ub, size=(self.x_dim,))
        else:
            self.ac_goal_pos = np.copy(INITIAL_GOAL_POS)

        logger.info("[SanityEnv] Resetting..\n\n")

        self._prev_obs = np.tile(self._obs[None], (self._obs_hist_len, 1))
        self._prev_act = np.zeros((self._act_hist_len, self.u_dim))

        return self.get_obs(), self.get_goal()

    def get_obs(self):
        obs = AttrDict(obs=self._obs[None].copy(),
                       prev_obs=self._prev_obs[None].copy(),
                       prev_act=self._prev_act[None].copy(),
                       latent=-np.ones((1, 1)))  # -1 specifies online
        return self._env_spec.map_to_types(obs)

    def get_goal(self):
        goal = AttrDict(goal_obs=np.tile(self.ac_goal_pos[None, None], (1, self.horizon + 1, 1)))
        return self._env_spec.map_to_types(goal)

    def get_cost(self):
        return np.abs(self._obs - self.ac_goal_pos).dot(self.np_weights)
