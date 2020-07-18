import mbrl.envs.ros_copter as ros_copter
import mbrl.envs.ros_copter_data_capture as ros_copter_dc

import numpy as np
from gym import spaces

from mbrl.experiments import logger
from mbrl.utils.mpc_utils import Metrics, Trajectory, SampleTrajectories
from mbrl.utils.python_utils import AttrDict, timeit

obs_mean = [0.41857021, 0.62113014, 0.0007459]
obs_std = [0.14432168, 0.13761707, 0.00088461]
WEIGHTS = [1 / s for s in obs_std]

action_std = [0.29055063, 0.45536894, 0.19936675]

XY_VELOCITY_MAX = 0.8  # 0.35 # limiting top speed artificially
Z_VELOCITY_MAX = 0.8  # 0.3 # limiting top speed artificially

WIDTH = 680
HEIGHT = 480

# PIX_TOL = [0.0075, 0.0075, 0.000175]
PIX_TOL = [0.01, 0.01, 0.00009]
PICKUP_PIX_TOL = [0.01, 0.01, 0.00005]

control_shapes = ['box', 'wait_box', 'circle', 'sine', 'center', 'figure8', 'slither',
                  'obstacle', 'follow']  # TODO fix follow, then add pickup


class TelloPendulumController:

    def __init__(self, params, env_spec):
        self._env_spec = env_spec

        # set up the parameters
        self._control = params.control.lower()  # trajectory to follow
        self._use_data_capture = params.use_data_capture
        self._use_future_goals = params.use_future_goals
        self._copter_params = params.copter_params
        self._copter_params['img_width'] = WIDTH
        self._copter_params['img_height'] = HEIGHT

        assert self._control in control_shapes

        # HARDCODED STUFF which eventually may change
        self.fix_waypoint3 = False
        self.no_ac1 = False

        # REGULAR STUFF
        if self._use_data_capture:
            print("### DATA CAPTURE ENV ###")
            self._copter = ros_copter_dc.RosCopterDataCapEnv(self._copter_params, self._env_spec)
        else:
            self._copter = ros_copter.RosCopterEnv(self._copter_params, self._env_spec)

        self.x_dim = self._env_spec.names_to_shapes.obs[-1]  # dimension of obs
        self.u_dim = self._env_spec.names_to_shapes.act[-1]  # dimension of act

        self._dt = self._copter_params.dt

        # artificial bounds, might be additionally constrained by env spec
        self.ac_lb = np.array([-XY_VELOCITY_MAX, -XY_VELOCITY_MAX, -Z_VELOCITY_MAX])
        self.ac_ub = np.array([XY_VELOCITY_MAX, XY_VELOCITY_MAX, Z_VELOCITY_MAX])
        self.action_space = spaces.Box(self.ac_lb, self.ac_ub)

        # allow for random sampling of the initial obs around the observed data
        self.observation_sample_mean = np.array(obs_mean)
        self.observation_sample_std = np.array(obs_std)

        self.action_sample_mean = np.zeros(len(action_std))
        self.action_sample_std = np.array(action_std)

        self.np_weights = np.array(WEIGHTS)
        self.cost_tol = np.array(PIX_TOL).dot(self.np_weights)  # (weighted cost)

        self.pets_computation_time = 0.1  # seconds

        # this determines when we move on
        wrap_func = lambda p, g: self._get_cost(p)
        self.waypoints = None
        self.metric = None
        self.reward_mask = None
        self.waiting_for_next = None
        self.loop = False

        if self._control == 'wait_box':
            # adaptive
            self.waypoints = [np.zeros((self.x_dim,)) for i in range(4)]
            self.waypoints[0][0] = 0.2
            self.waypoints[0][1] = 0.6
            self.waypoints[0][2] = 0.0007459
            self.waypoints[1][0] = 0.2
            self.waypoints[1][1] = 0.2
            self.waypoints[1][2] = 0.0007459
            self.waypoints[2][0] = 0.7
            self.waypoints[2][1] = 0.2
            self.waypoints[2][2] = 0.0007459
            self.waypoints[3][0] = 0.7
            self.waypoints[3][1] = 0.6
            self.waypoints[3][2] = 0.0007459

            self.waypoints = iter(self.waypoints)
            self.metric = Metrics.ext_function_thresh(self.cost_tol, wrap_func)
            self.reward_mask = lambda pos, i: -self._get_cost(pos) if i > 1 else 0
            self.waiting_for_next = lambda i: True

        elif self._control == 'box':
            # not adaptive, iterates along sides
            len_x = len_y = 0.5
            num_pts_per_side = 14  # 3.5 seconds
            center = np.array([0.5, 0.5, 0.0007459])
            inc_x = len_x / num_pts_per_side
            inc_y = len_y / num_pts_per_side
            self.waypoints = [center + np.array([len_x / 2.0, len_y / 2.0, 0])]  # start

            # clockwise
            for i in range(num_pts_per_side):
                self.waypoints += [self.waypoints[-1] + np.array([-inc_x, 0, 0])]
            for i in range(num_pts_per_side):
                self.waypoints += [self.waypoints[-1] + np.array([0, -inc_y, 0])]
            for i in range(num_pts_per_side):
                self.waypoints += [self.waypoints[-1] + np.array([inc_x, 0, 0])]
            for i in range(num_pts_per_side):
                self.waypoints += [self.waypoints[-1] + np.array([0, inc_y, 0])]

            self.metric = Metrics.wait_start_sequential(Metrics.ext_function_thresh(self.cost_tol, wrap_func))
            self.reward_mask = lambda pos, i: -self._get_cost(pos) if i > 1 else 0
            self.waiting_for_next = lambda i: False if i > 1 else True  # waits only for first point

        elif self._control == 'circle':
            # 12.5 seconds (TODO CHANGE BACK!! to 70)
            self.waypoints = SampleTrajectories.circle_trajectory((0.5, 0.35, 0.0007459), ax1=0, ax2=1, radius=0.25,
                                                                  num_points_per_rot=50, num_rot=1)  # 70
            self.metric = Metrics.wait_start_sequential(Metrics.ext_function_thresh(self.cost_tol, wrap_func))
            self.reward_mask = lambda pos, i: -self._get_cost(pos) if i > 1 else 0
            self.waiting_for_next = lambda i: False if i > 1 else True  # waits only for first point

        elif self._control == 'sine':
            # 20 seconds
            self.waypoints = SampleTrajectories.sinusoid_trajectory((0.5, 0.5, 0.0007459), ax=0, amp=0.35,
                                                                    num_points_per_rot=40, num_rot=2)
            self.metric = Metrics.wait_start_sequential(Metrics.ext_function_thresh(self.cost_tol, wrap_func))
            self.reward_mask = lambda pos, i: -self._get_cost(pos) if i > 1 else 0
            self.waiting_for_next = lambda i: False if i > 1 else True  # waits only for first point

        elif self._control == 'slither':
            # 20 seconds
            self.waypoints = SampleTrajectories.slither_trajectory((0.5, 0.5, 0.0007459), ax=0, amp=0.35, ax_vert=1,
                                                                   vert_amp=0.25, num_points_per_rot=30, num_rot=3)
            self.metric = Metrics.wait_start_sequential(Metrics.ext_function_thresh(self.cost_tol, wrap_func))
            self.reward_mask = lambda pos, i: -self._get_cost(pos) if i > 1 else 0
            self.waiting_for_next = lambda i: False if i > 1 else True  # waits only for first point

        elif self._control == 'figure8':
            # 20 seconds=
            self.waypoints = SampleTrajectories.figure8_trajectory(0.5, 0.5, np.sqrt(0.0009), x_amp=0.3, z_factor=0.35,
                                                                   num_points_per_rot=80, num_rot=1)
            self.metric = Metrics.wait_start_sequential(Metrics.ext_function_thresh(self.cost_tol, wrap_func))
            self.reward_mask = lambda pos, i: -self._get_cost(pos) if i > 1 else 0
            self.waiting_for_next = lambda i: False if i > 1 else True  # waits only for first point

        elif self._control == 'center':
            self.waypoints = [np.array([0.5, 0.5, 0.0007459])]

            self.waypoints = iter(self.waypoints)
            self.metric = Metrics.ext_function_thresh(self.cost_tol, wrap_func)
            self.reward_mask = lambda pos, i: -self._get_cost(pos)
            self.waiting_for_next = lambda i: True

        elif self._control == 'follow':
            self.waypoints = [np.array([0.5, 0.5, 0.001])]  # 0.0007459])]

            self.waypoints = iter(self.waypoints)
            self.metric = Metrics.ext_function_thresh(self.cost_tol, wrap_func)
            self.reward_mask = lambda pos, i: -self._get_cost(pos)
            self.waiting_for_next = lambda i: True
            self.loop = True

        elif self._control == 'follow_pickup':
            self.waypoints = [np.array([0.5, 0.5, 0.001])]  # 0.0007459])]

            self.waypoints = iter(self.waypoints)
            self.metric = Metrics.ext_function_thresh(self.cost_tol, wrap_func)
            self.reward_mask = lambda pos, i: -self._get_cost(pos)
            self.waiting_for_next = lambda i: True

        elif self._control == 'obstacle':
            base = 0.0006
            second = 0.0017
            self.waypoints = [np.array([0.15, 0.6, base])]
            self.waypoints += [np.array([0.2, 0.6, base])]
            self.waypoints += [np.array([0.23, 0.6, second])]
            self.waypoints += [np.array([0.5, 0.6, second])]
            self.waypoints += [np.array([0.67, 0.6, second])]
            self.waypoints += [np.array([0.8, 0.6, base])]
            self.waypoints += [np.array([0.9, 0.6, base])]

            self.waypoints = SampleTrajectories.even_interpolate(self.waypoints, 8)

            self.waypoints = iter(self.waypoints)
            self.metric = Metrics.wait_start_sequential(Metrics.ext_function_thresh(self.cost_tol, wrap_func))
            self.reward_mask = lambda pos, i: -self._get_cost(pos) if i > 1 else 0
            self.waiting_for_next = lambda i: False if i > 1 else True  # waits only for first point

        else:
            raise NotImplementedError

        self.trajectory = Trajectory('absolute',
                                     self.waypoints,
                                     self.metric,  # moves on when func(obs, goal) < thresh
                                     self.waiting_for_next,
                                     np.zeros((self.x_dim,)),
                                     loop=self.loop)

        self.reset_model()

    # this weight default comes from the data (normalizing)
    def _get_cost(self, pos):
        # 10 comes from the square term
        return np.abs(pos - self._copter._curr_goal_pos).dot(self.np_weights)

    def reset_model(self):
        self.trajectory.reset()

        obs = self.get_obs()

        self._copter.set_target(self.trajectory.next(obs.obs[0]), self.fix_waypoint3)
        return obs

    def target_in_frame(self):
        obs = self.get_obs().obs[0]
        return not (obs[0] < 1e-4 and obs[1] < 1e-4)

    def reset(self):
        # this will take off the model if online
        self._copter.reset()
        # we need to wait til the observation is nonzero
        if not self._use_data_capture and not self._copter.offline:  # only wait if not in data capture mode
            logger.info("[TELLO CTRL] Waiting for target to be in frame...")
            while not self.target_in_frame():
                pass
            logger.info("[TELLO CTRL] Target is in frame, press enter to start the rollout: ")
            input()
            logger.info("[TELLO CTRL] Starting the rollout...")

        return self.reset_model(), self.get_goal()

    def _step(self, action, **kwargs):

        with timeit("copter_step", reset_on_stop=True):
            obs, _, done = self._copter.step(action, no_rew_pub=True, **kwargs)

        pos = obs.obs[0]
        true_reward = -self._get_cost(pos)
        reward = self.reward_mask(pos, self.trajectory.get_i())

        # TODO
        self._copter.pub_rew(rx=true_reward,
                             ry=reward)  # publish the true reward and masked reward for the sake of accurate data collect

        # print(pos[:3], self.ac_goal_pos[:3], self.ac_goal_pos_index)
        # print('reward', reward)
        # print(action, np.array2string(pos, separator=', '), self._copter._curr_goal_pos, reward, true_reward, done,
        #       self.trajectory.get_i(), self.trajectory.curr_goal)

        self._copter.set_target(self.trajectory.next(pos), self.fix_waypoint3)

        done = done or self.trajectory.is_finished()  # returns true when the full trajectory has been run through

        ### online control safety measures

        # stop if out of frame
        if pos[0] < 1e-4 and pos[1] < 1e-4:
            # target is out of frame
            done = True
            logger.warn("[TELLO CTRL] Target Left frame! Terminating rollout")

        # if terminating, send stop signal
        if self._use_data_capture and not self._copter.offline and done:
            self._copter.sig_rollout(end=True)

        goal = self.get_goal()

        def arr3D_2str(arr, names=("x", "y", "z")):
            str = "{%s: %.5f, %s: %.5f, %s: %.5f }" \
                  % (names[0], arr[0], names[1], arr[1], names[2], arr[2])
            return str

        logger.debug("[TPC] T: %.4f sec // OBS: %s -- GOAL: %s -- ACT: %s" %
                     (timeit.elapsed("copter_step"), arr3D_2str(pos), arr3D_2str(goal.goal_obs[0, 0]), arr3D_2str(action.act[0])))

        return obs, goal, done

    def step(self, action, **kwargs):
        return self._step(action, **kwargs)

    def get_obs(self):
        return self._copter.get_obs()

    # (1, H+1, dim1..)
    def get_goal(self):
        if self._use_future_goals and self._copter.horizon > 0:
            goal = self._copter.get_goal().goal_obs[0, 0]
            next_n = self.trajectory.try_next_n(self._copter.horizon)\
                .reshape(self._copter.horizon, self.x_dim)
            future_goals = np.concatenate([goal[None], next_n], axis=0)
            return self._env_spec.map_to_types(AttrDict(goal_obs=future_goals[None]))
        else:
            return self._copter.get_goal()

