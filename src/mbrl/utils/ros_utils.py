import threading
from threading import Lock

import numpy as np
from dotmap import DotMap
import rospy

from sensor_msgs.msg import Joy
from crazyflie.msg import CFCommand
from geometry_msgs.msg import Vector3Stamped

from mbrl.utils.mpc_utils import Trajectory


class RosTopic:
    def __init__(self, topic, type):
        self._topic = topic
        self._type = type

    @property
    def topic(self):
        return self._topic

    @property
    def type(self):
        return self._type

    def __eq__(self, other):
        return other.topic == self.topic and other.type == self.type

    def __hash__(self):
        return hash("%s %s" % (self._topic, self._type))


############ SINGLE ROS INSTANCE ###########

_ros_initialized = False
_ros_topic_pub_map = dict()
_ros_topic_sub_list = []
_rate = None
_background_thread = None


#  called once
def setup_ros_node(name, **kwargs):
    global _ros_initialized
    assert not _ros_initialized, 'ROS was already intialized'
    rospy.init_node(name, **kwargs)
    _ros_initialized = True


#  returns tuple (was_created, publisher)
def get_publisher(rt: RosTopic, **pub_kwargs):
    created = False
    if rt not in _ros_topic_pub_map.keys():
        created = True
        _ros_topic_pub_map[rt] = rospy.Publisher(rt.topic, rt.type, **pub_kwargs)

    return created, _ros_topic_pub_map[rt]


#  returns bound subscriber, may need to pass in callback_args or queue_size
def bind_subscriber(rt: RosTopic, sub_cb, **sub_kwargs):
    sub = rospy.Subscriber(rt.topic, rt.type, sub_cb, **sub_kwargs)
    _ros_topic_sub_list.append(DotMap({"top": rt, "sub": sub}))
    return sub


# calls loop function(rate) periodically as determined by rate in separate thread
def setup_background_thread(loop_fn, rate):
    global _rate
    _rate = rate

    def _thread_fn():
        while not rospy.is_shutdown():
            # inner loop to publish stationary messages in between rollouts
            loop_fn()
            _rate.sleep()

    global _background_thread
    _background_thread = threading.Thread(target=_thread_fn)


# begins background thread
def start_background_thread():
    assert _background_thread is not None, "No background thread instantiated"
    _background_thread.start()


class TrajectoryRos(Trajectory):

    # type can be absolute, relative, or velocity
    # metric represents evaluation
    def __init__(self, update_func, topic_type_list, traj_type, iterator, metric, is_waiting_func, origin, loop=False):
        super().__init__(traj_type, iterator, metric, is_waiting_func, origin, loop=loop)

        self.do_update = False
        self.update_func = update_func
        self.topic_type_list = list(topic_type_list)  # list of tuples (topic, type)

        def update_if(msg):
            if self.do_update:
                update_func(self, msg)  # traj object,msg

        self.subs = [rospy.Subscriber(topic, tp, update_if) for topic, tp in self.topic_type_list]

    def next(self, state):
        out = super().next(state)
        self.do_update = True
        if self.completed:
            self.do_update = False
        return out

    def reset(self, **kwargs):
        self.do_update = False
        super().reset()


# pickup trajectory with camera follow
def camera_follow_trajectory(latent_reset_func, waypoints, metric, waiting_for_next, origin, loop=False):

    LRESET_BUTTON = 8
    _joy_msg = Joy()

    def update_func(traj_obj, msg):
        nonlocal _joy_msg

        if type(msg) == Joy:
            if msg.buttons[LRESET_BUTTON] and not _joy_msg.buttons[LRESET_BUTTON]:
                # RESET
                latent_reset_func()
            _joy_msg = msg
        else:
            pass

    traj = TrajectoryRos(update_func, [('/joy', Joy)],
                         'absolute',
                         waypoints,
                         metric,  # moves on when func(obs, goal) < thresh
                         waiting_for_next,
                         origin, loop=loop)

    return traj
