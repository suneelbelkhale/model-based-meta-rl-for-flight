"""
The trainer is where everything is combined.
"""
import math
import shutil

from mbrl.experiments import logger
from mbrl.models.latent_model import LatentModel
from mbrl.trainers.latent_trainer import LatentTrainer, is_next_cycle
from mbrl.utils.python_utils import timeit, AttrDict
from dotmap import DotMap as d
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import csv


# trains the latent variable during inference
from mbrl.utils.torch_utils import to_torch, to_numpy


class LatentInferenceTrainer(object):

    def __init__(self, params, model: LatentModel, dataset_inference):
        # self._file_manager = file_manager
        self._model = model
        self._dataset_inference = dataset_inference
        # self._base_optimizer = optim.Adam(model.base_parameters(), lr=float(params.dynamics_learning_rate))
        self._latent_optimizer = optim.Adam(model.latent_parameters(), lr=float(params.latent_learning_rate))

        self._train_min_buffer_size = max(int(params.train_min_buffer_size), 2)  # after N steps, do latent training
        self._log_every_n_steps = int(params.log_every_n_steps)  # must be a multiple of _train_every_n_steps
        self._save_every_n_steps = int(params.save_every_n_steps)  # must be a multiple of _train_every_n_steps # TODO
        self._train_every_n_steps = int(params.train_every_n_steps)

        self._max_steps_per_rollout = int(params.get("max_steps_per_rollout", 0))

        self._obs_to_output_obs_fn = params.obs_to_output_obs_fn

        self._step = 0

        self._current_latent_step = 0
        self._current_latent_loss = math.inf

        self._exp_avg_steps_per_train = 0
        self._alpha = 0.5  # how much of the new sample to use

        self._reset_curr_episode()

    def _reset_curr_episode(self):
        self._dataset_inference.reset()
        self._model.reset_latent_model()
        self._step = 0

    def _log(self):
        logger.info('[Latent Training] Step: {}, Loss: {}, Steps/train (avg): {}'
                    .format(self._current_latent_step, self._current_latent_loss, self._exp_avg_steps_per_train))
        mu, lsig = self._model.get_online_latent_mu_logsig()
        logger.debug('[Latent Training] MU: %s, SIGMA: %s' % (to_numpy(mu).tolist(), to_numpy(lsig.exp()).tolist()))

    # trains latent until an exit condition returns true (blocking), then returns current latent distribution
    # only uses at most buffer_window_size
    def _train_latent(self, exit_cond, buffer_window_size=-1):
        # just to check we have enough data points
        assert self._step <= len(self._dataset_inference)

        # train on the right cycle
        if not is_next_cycle(self._step, self._train_every_n_steps):
            while not exit_cond(0):
                pass
            return self._model.get_online_latent_mu_logsig()

        # restrict training to some time frame
        if buffer_window_size == -1:
            buffer_window_size = self._step
        buffer_window_size = min(buffer_window_size, self._step)
        dataset_min_idx = len(self._dataset_inference) - buffer_window_size

        # must have at least 2 data points in buffer window to begin latent training
        if buffer_window_size < self._train_min_buffer_size:
            return self._model.get_online_latent_mu_logsig()

        i = 0
        while not exit_cond(i):
            # inner-most loop
            inputs, outputs = self._dataset_inference.get_batch(torch_device=self._model.device, min_idx=dataset_min_idx)
            self._model.train()
            latent_loss = self._model.latent_loss(inputs, outputs)
            self._latent_optimizer.zero_grad()
            latent_loss.backward()
            self._latent_optimizer.step()
            self._current_latent_loss = latent_loss.item()
            self._current_latent_step += 1
            i += 1

            if is_next_cycle(self._current_latent_step, self._log_every_n_steps):
                self._log()

        # moving average
        self._exp_avg_steps_per_train = self._alpha * i + (1 - self._alpha) * self._exp_avg_steps_per_train

        return self._model.get_online_latent_mu_logsig()

    def step(self, env, obs, goal, action):
        next_obs, next_goal, done = env.step(action, step_extra_func=self._train_latent)
        self._step += 1

        # optional forced reset
        if 1 < self._max_steps_per_rollout <= self._step:
            done = True

        next_out = self._obs_to_output_obs_fn(next_obs.leaf_apply(lambda arr: to_torch(arr, check=True)))
        # add sample to dataset
        self._dataset_inference.add_sample(
            obs.leaf_apply(lambda arr: to_numpy(arr, check=True)),
            next_out.leaf_apply(lambda arr: to_numpy(arr, check=True)),
            goal.leaf_apply(lambda arr: to_numpy(arr, check=True)),
            action.leaf_apply(lambda arr: to_numpy(arr, check=True)),
            np.bool(done)
        )

        if done:
            self._reset_curr_episode()
            next_obs, next_goal = env.reset()

        return next_obs, next_goal

