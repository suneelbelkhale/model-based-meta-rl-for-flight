from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import rospy
from crazyflie.utils import ServantSigController, STOP, START

import numpy as np
import math
import time

# online environment
from mbrl.envs.ros_copter import RosCopterEnv
from mbrl.experiments import logger
from mbrl.utils import abstract


class RosCopterDataCapEnv(RosCopterEnv):
    def __init__(self, params, env_spec):
        self._first_step = True
        self._servant = None
        self._ext_stop = False

        # calls reset
        super().__init__(params, env_spec)

    @abstract.overrides
    def _bg_loop_fn(self):
        pass

    @abstract.overrides
    def _step(self, u, no_rew_pub=False, step_extra_func=None):
        if self._first_step:
            self._servant.send_sig(START)  # acknowledges we received the start
            self._first_step = False

        super()._step(u, no_rew_pub=no_rew_pub, step_extra_func=step_extra_func)

        # externally triggered stop, master controller waits for ack
        if self._servant.sender_stopped():
            logger.info("[CF] STOPPED EXTERNALLY.")
            self.done = True

        if self._ext_stop:
            logger.info("[CF] STOPPED INTERNALLY.")
            self.done = True

        if self.done:
            self._servant.send_sig(STOP)

        return self.get_obs(), self.get_goal(), self._done

    @abstract.overrides
    def sig_rollout(self, end):
        assert end  # must be a stop command
        self._ext_stop = end
        self._servant.send_sig(STOP)  # preemptively send

    # this will get called when we receive a STOP
    @abstract.overrides
    def reset(self):

        if self._servant is None:
            self._servant = ServantSigController(self._ros_prefix)

        logger.info("[CF]: Resetting.")

        self._servant.reset_sig()
        # self.servant.send_sig(STOP) # acknowledges we received the stop (or triggers a data collection stop)

        logger.info("[CF]: Waiting for Master Controller start before continuing")
        # now we wait for a start before continuing
        self._servant.wait_sig(START, rate=rospy.Rate(10))

        self._servant.reset_sig()

        self.collided = False
        self._ext_stop = False
        self._first_step = True
        return self.reset_model(), self.get_goal()
