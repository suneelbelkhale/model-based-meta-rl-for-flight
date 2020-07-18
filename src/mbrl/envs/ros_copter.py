import time

from mbrl.experiments import logger
from mbrl.utils import abstract
import mbrl.utils.ros_utils as ru
import numpy as np

from mbrl.utils.np_utils import advance_history_np
from mbrl.utils.python_utils import AttrDict
from mbrl.utils.ros_utils import RosTopic

import rospy
from crazyflie.msg import CFData
from crazyflie.msg import CFMotion
from crazyflie.msg import CFCommand
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped, Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from visualization_msgs.msg import Marker

from gym import spaces

# NOTE: instantiating this class requires ROS
from mbrl.utils.torch_utils import to_numpy


class RosCopterEnv(abstract.BaseClass):

    def __init__(self, params, env_spec):
        env_spec.assert_has_names(['obs', 'act', 'prev_obs', 'prev_act'])

        self._env_spec = env_spec

        self.x_dim = self._env_spec.names_to_shapes.obs[-1]  # dimension of obs
        self.u_dim = self._env_spec.names_to_shapes.act[-1]  # dimension of act

        self._obs_hist_len = self._env_spec.names_to_shapes.prev_obs[-2]
        self._act_hist_len = self._env_spec.names_to_shapes.prev_act[-2]

        # Setup the parameters
        self._dt = params.dt
        self._ros_prefix = params.ros_prefix

        self.offline = params.offline  # running offline means don't publish anything (e.g. from rosbag)
        self.horizon = int(params.horizon)

        self._normalize = params.normalize

        if self._normalize:
            logger.debug("Normalizing input ON")
            self._coef_cx = 1. / params.img_width
            self._coef_cy = 1. / params.img_height
            self._coef_area = 1. / (params.img_width * params.img_height)
        else:
            self._coef_cx = self._coef_cy = self._coef_area = 1.

        self.time_last_step = -1

        self._obs = np.zeros((self.x_dim,))
        self._next_obs = np.zeros((self.x_dim,))
        self._prev_obs = None
        self._prev_act = None
        self._ob_lb, self._ob_ub = self._env_spec.limits(['obs'])
        self._ob_lb, self._ob_ub = self._ob_lb[0], self._ob_ub[0]
        self.observation_space = spaces.Box(self._ob_lb, self._ob_ub)

        self._initial_goal_pos = params.initial_goal_pos
        self._curr_goal_pos = None

        self._cost = 0
        self._done = False  # indicates if the copter has completed/failed its rollout
        self._running = False  # indicates if a rollout is currently running

        # start with an assumption of collision to allow for takeoff
        self.collided = True

        self._prev_act = None
        self._ac_lb, self._ac_ub = self._env_spec.limits(['act'])
        self._ac_lb, self._ac_ub = self._ac_lb[0], self._ac_ub[0]
        self.action_space = spaces.Box(self._ac_lb, self._ac_ub)

        # ros callbacks
        self._curr_motion = CFMotion()

        self._ros_topics_and_types = {
            self._ros_prefix + 'coll': Bool,
            self._ros_prefix + 'data': CFData,
            'extcam/target_vector': Vector3Stamped,
        }

        self._ros_msgs = dict()
        self._ros_msg_times = dict()

        # ROS setup
        ru.setup_ros_node('MBRLNode', anonymous=True)
        for topic, type in self._ros_topics_and_types.items():
            ru.bind_subscriber(RosTopic(topic, type), self.ros_msg_update, callback_args=(topic,))

        # does not use these publishers if _offline
        _, self._ros_motion_pub = ru.get_publisher(RosTopic(self._ros_prefix + "motion", CFMotion), queue_size=10)
        _, self._ros_command_pub = ru.get_publisher(RosTopic(self._ros_prefix + "command", CFCommand), queue_size=10)
        _, self._ros_goal_pub = ru.get_publisher(RosTopic("/mpc/goal_vector", Vector3Stamped), queue_size=10)
        _, self._ros_action_pub = ru.get_publisher(RosTopic("/mpc/action_vector", Vector3Stamped), queue_size=10)
        _, self._ros_reward_pub = ru.get_publisher(RosTopic("/mpc/reward_vector", Vector3Stamped), queue_size=10)
        _, self._ros_latent_pub = ru.get_publisher(RosTopic("/mpc/latent_vector", Vector3Stamped), queue_size=10)

        _, self._ros_trajectory_marker_pub = ru.get_publisher(RosTopic("/mpc/trajectory_marker", Marker), queue_size=500)
        _, self._ros_ac_marker_pub = ru.get_publisher(RosTopic("/mpc/action_marker", Marker), queue_size=10)

        logger.info("[COPTER]: Initialized, waiting for topic streams before starting...")
        while not self.ros_is_good(display=False):
            pass
        logger.info("[COPTER]: Topics are active.")

        self._step_rate = rospy.Rate(1. / self._dt)  # used in step extra function
        self._rate = rospy.Rate(1. / self._dt)  # used only in background thread

        # this background publisher thread is only when we are online
        if not self.offline:
            ru.setup_background_thread(self._bg_loop_fn, self._rate)
            ru.start_background_thread()

        self.reset()

    # sends action messages when model has been reset
    def _bg_loop_fn(self):
        if not self._running:
            # publish steady hold message when asking for zero motion
            motion = CFMotion()
            motion.is_flow_motion = True
            motion.stamp.stamp = rospy.Time.now()
            self._ros_motion_pub.publish(motion)

    def _step(self, u, no_rew_pub=False, step_extra_func=None, buffer_window_size=-1, **kwargs):
        self._running = True  # signifies that we are publishing actions now

        if not self.offline:
            assert self.ros_is_good(display=True)
        # else:
        #     u = self._latest_offline_ac.copy()

        obs = self._next_obs.copy()

        self._enforce_dimension(obs, u)

        # Limit the control inputs
        u0 = np.clip(u, self._ac_lb, self._ac_ub)

        # update vx, vy, dz (does NOT publish)
        self._set_motion(u0[0], u0[1], u0[2])

        self._done = np.all(obs == self._curr_goal_pos) or self.collided

        if np.any(np.isnan(obs)):
            logger.debug('[CF]: NAN POSITION')

        # publishing action messages happens only if we are fully online
        if not self.offline:
            if any((self._curr_motion.x, self._curr_motion.y, self._curr_motion.yaw,

                    self._curr_motion.dz)) or not self.collided:
                self._curr_motion.stamp.stamp = rospy.Time.now()
                self._ros_motion_pub.publish(self._curr_motion)
            else:
                # publish steady hold message when asking for zero motion
                motion = CFMotion()
                motion.is_flow_motion = True
                motion.stamp.stamp = rospy.Time.now()
                self._ros_motion_pub.publish(motion)
        else:
            # offline action publishing
            act_vec = Vector3Stamped()
            act_vec.header.stamp = rospy.Time.now()
            act_vec.vector.x = self._curr_motion.x
            act_vec.vector.y = self._curr_motion.y
            act_vec.vector.z = self._curr_motion.dz
            self._ros_action_pub.publish(act_vec)

        goal_vec = Vector3Stamped()
        goal_vec.header.stamp = rospy.Time.now()
        goal_vec.vector.x = self._curr_goal_pos[0]
        goal_vec.vector.y = self._curr_goal_pos[1]
        goal_vec.vector.z = self._curr_goal_pos[2]
        self._ros_goal_pub.publish(goal_vec)

        # TODO: set cost properly and in a single place
        if not no_rew_pub:
            self.pub_rew(-self._cost)

        ## EXTRA FUNCTIONALITY for inferring latent online
        ret = None
        if step_extra_func is not None:
            def exit_condition(iters): return self._step_rate.remaining().to_sec() < 1e-7 \
                                           and self._step_rate.sleep() is None
            ret = step_extra_func(exit_cond=exit_condition,
                                  buffer_window_size=buffer_window_size)  # returns at or before sleep_end_time
            mu = ret[0][0].item()
            sig = ret[1][0].exp().item()  # TODO do something with these
            latent_vec = Vector3Stamped()
            latent_vec.header.stamp = rospy.Time.now()
            latent_vec.vector.x = mu
            latent_vec.vector.y = sig
            self._ros_latent_pub.publish(latent_vec)
        else:
            self._step_rate.sleep()  # sleep until next iter

        # history update
        self._prev_obs = advance_history_np(self._prev_obs[None], obs[None])[0]
        self._prev_act = advance_history_np(self._prev_act[None], u0[None])[0]

        self._next_obs = self._obs.copy()  # the exact obs the policy sees
        return self.get_obs(), self.get_goal(), self._done

    # @TODO
    def ros_msg_update(self, msg, args):
        topic = args[0]

        if 'coll' in topic:
            # when collision is True
            if msg.data:
                self.collided = True
                self._done = True
        elif 'data' in topic:
            pass
        elif 'target_vector' in topic:
            # update state
            self._obs[0] = msg.vector.x * self._coef_cx  # cz
            self._obs[1] = msg.vector.y * self._coef_cy  # cy
            self._obs[2] = msg.vector.z * self._coef_area  # area
        else:
            logger.warn("[CF]: Unhandled ROS msg")

        self._ros_msgs[topic] = msg
        self._ros_msg_times[topic] = rospy.Time.now()

    def ros_is_good(self, display):
        for topic in self._ros_topics_and_types.keys():
            # make sure we have received at least one msg of each type recently
            if 'coll' not in topic:
                if topic not in self._ros_msg_times:
                    if display:
                        logger.warn('Topic {0} has never been received'.format(topic))
                    return False
                elapsed = (rospy.Time.now() - self._ros_msg_times[topic]).to_sec()
                if elapsed > 2:  # self._dt * 50:
                    if display:
                        logger.warn(
                            'Topic {0} was received {1} seconds ago (dt is {2})'.format(topic, elapsed, self._dt))
                    return False
        return True

    def pub_rew(self, rx, ry=0, rz=0):
        reward = Vector3Stamped()
        reward.vector.x = rx
        reward.vector.y = ry
        reward.vector.z = rz
        reward.header.stamp = rospy.Time.now()
        self._ros_reward_pub.publish(reward)

    def pub_trajectories(self, full_action, sample_n=10):
        acts = full_action.action_sequence.act[0]  # (1, P, horizon, actdim)
        order = full_action.results.order[0]  # (1, P) -> (P,)
        costs = full_action.results.costs[0]  # (1, P,)
        traj = full_action.results.trajectory.obs[0]  # (1, P, H+1, obsdim)

        assert acts.shape[0] == order.shape[0] == costs.shape[0] == traj.shape[0]

        assert self.horizon == acts.shape[1] == traj.shape[1] - 1

        if sample_n == -1:
            sample_n = acts.shape[0]
        sample_n = min(sample_n, acts.shape[0])
        inc = max(1, acts.shape[0] // sample_n)

        # for each trajectory, send it as a Marker
        for i in range(sample_n):
            idx = order[i * inc]
            this_as = acts[idx]
            this_tr = traj[idx]
            this_c = costs[idx]

            m = Marker()
            m.header.stamp = rospy.Time.now()
            m.header.frame_id = "world"
            m.ns = 'mpc_vis';  m.id = i;  m.type = Marker.LINE_STRIP
            m.lifetime = rospy.Duration()
            m.scale.x = 0.05
            m.color.r = 0.4;  m.color.g = 0.4;  m.color.b = 0.4;  m.color.a = 1.0

            # this is how we pass the cost
            m.text = "%f" % this_c

            m.pose.position.x = this_tr[0, 0]
            m.pose.position.y = this_tr[0, 1]
            m.pose.position.z = this_tr[0, 2]

            m.pose.orientation.w = 1

            for j in range(self.horizon + 1):
                if this_tr.shape[1] == 2:
                    m.points.append(Point(this_tr[j, 0], this_tr[j, 1], 0.))
                else:
                    m.points.append(Point(this_tr[j, 0], this_tr[j, 1], this_tr[j, 2]))

            self._ros_trajectory_marker_pub.publish(m)

        # send the best action sequence too
        acm = Marker()
        acm.header.stamp = rospy.Time.now()
        acm.header.frame_id = "cf"
        acm.text = "%f" % costs[order[0]]
        acm.type = Marker.LINE_STRIP;  acm.lifetime = rospy.Duration()
        for j in range(self.horizon):
            ac = acts[order[0], j, :3].tolist()
            ac += [0] * (3 - len(ac))  # padding
            acm.points.append(Point(*ac))

        self._ros_ac_marker_pub.publish(acm)

    def is_takeoff_ready(self):
        # checks that we aren't upside_down
        alt = self._ros_msgs[self._ros_prefix + 'data'].alt
        if alt > 0.05:
            return False
        return True

    # externally toggle _running
    def sig_rollout(self, end):
        self._running = not end

    def _set_motion(self, vx, vy, vz):
        motion = CFMotion()
        motion.x = vx
        motion.y = vy
        motion.yaw = 0  # yaw disabled for now
        motion.dz = vz  # convert velocity to a dt for this time step
        motion.is_flow_motion = True

        self._curr_motion = motion

    def _set_command(self, cmd):
        # 0 is ESTOP, 1 IS LAND, 2 IS TAKEOFF
        command = CFCommand()
        command.cmd = cmd
        self._ros_command_pub.publish(command)

    def set_target(self, xf, fix_waypoint3=False):
        self._curr_goal_pos = xf
        if fix_waypoint3:
            self._curr_goal_pos[2] = self._obs[2]

    def _set_target_relative(self, xf, fix_waypoint3=False):
        self._curr_goal_pos = xf + self._obs
        if fix_waypoint3:
            self._curr_goal_pos[2] = self._obs[2]

    def _enforce_dimension(self, x, u):
        assert x.shape[-1] == self.x_dim and u.shape[-1] == self.u_dim

    def step(self, action, no_rew_pub=False, **kwargs):
        # convert to numpy to be safe
        action = action.leaf_apply(lambda tin: to_numpy(tin, check=True))
        self.pub_trajectories(full_action=action)
        return self._step(action.act[0], no_rew_pub=no_rew_pub, **kwargs)

    def reset(self):
        self._running = True  # stops background thread
        logger.info("[CF]: Resetting.")
        # means we ended on a collision
        if not self.offline and self.collided:
            # ensure we are on the ground
            self._set_command(0)
            logger.info("[CF]: Estop CMD Sent.")
            time.sleep(1)

            # block if we are waiting for takeoff readiness
            if not self.is_takeoff_ready():
                logger.info("[CF]: Takeoff is not ready. Press Enter to Continue:")
                input()

            # takeoff
            self._set_command(2)
            logger.info("[CF]: Takeoff CMD Sent.")
            time.sleep(3)

        self.collided = False
        self._running = False  # starts background thread to keep steady while waiting for commands

        return self.reset_model(), self.get_goal()

    def reset_model(self):
        self._curr_goal_pos = np.copy(self._initial_goal_pos)
        self._prev_obs = np.tile(self._obs[None], (self._obs_hist_len, 1))
        self._prev_act = np.zeros((self._act_hist_len, self.u_dim))
        self._next_obs = self._obs.copy()

        return self.get_obs()

    def get_obs(self):
        obs = AttrDict(obs=self._obs[None].copy(),
                       prev_obs=self._prev_obs[None].copy(),
                       prev_act=self._prev_act[None].copy(),
                       latent=-np.ones((1, 1)))  # -1 specifies online
        return self._env_spec.map_to_types(obs)

    def get_goal(self):
        goal = AttrDict(goal_obs=np.tile(self._curr_goal_pos[None, None], (1, self.horizon + 1, 1)))
        return self._env_spec.map_to_types(goal)

    # def _get_cost(self):
    #     return np.sum(np.absolute(self._obs - self._curr_goal_pos))
