from typing import Callable

import numpy as np
import scipy.io
import torch
import os

from mbrl.datasets.dataset import Dataset
from mbrl.experiments import logger
from mbrl.experiments.file_manager import FileManager
from mbrl.utils import abstract, file_utils, np_utils
from mbrl.utils.data_utils import split_data_by_episodes
from mbrl.utils.python_utils import AttrDict


class MatDataset(Dataset):

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        self._input_file = params.input_file  # None if we are starting a new file
        self._output_file = params.output_file  # relative path in exp dir
        self._batch_size = params.batch_size  # number of samples to get per training epoch
        self._planning_horizon = params.planning_horizon  # obs sequence length per batch item
        self._obs_history_length = params.obs_history_length  # obs history in model input
        self._acs_history_length = params.acs_history_length  # action history in model input
        # self._save_every_n_steps = params.save_every_n_steps
        # assert self._save_every_n_steps == 0 or self._output_file

        self._all_names = self._env_spec.observation_names + \
                          self._env_spec.action_names + \
                          self._env_spec.output_observation_names + \
                          self._env_spec.goal_names + ['done']

        self._data_len = 0
        self._num_episodes = 0
        self._split_indices = np.array([])

        o_shape = self._env_spec.names_to_shapes.obs
        self._sigma_obs = np.ones(o_shape)
        self._mu_obs = np.zeros(o_shape)

    @abstract.overrides
    def _init_setup(self):
        self._initial_datadict = self._load_mat()
        self._initial_split_indices = self._split_indices.copy()
        self._initial_data_len = self._data_len
        self.reset()

    def _load_mat(self):
        local_dict = AttrDict()

        if self._input_file is None:
            local_dict = self._env_spec.get_zeros(self._all_names, 0)  # np
        else:
            logger.debug('Loading ' + self._input_file)
            samples = scipy.io.loadmat(self._input_file)

            # split into chunks by episode. dict = {key: list of [Ni, key_shape]}
            data_dict = split_data_by_episodes(samples,
                                               horizon=self._planning_horizon,
                                               n_obs=self._obs_history_length,
                                               n_acs=self._acs_history_length)

            self._mu_obs = data_dict['mu_obs']
            self._sigma_obs = data_dict['sigma_obs']
            self._mu_delta_obs = data_dict['mu_delta_obs']
            self._sigma_delta_obs = data_dict['sigma_delta_obs']

            self._action_sequences = np.concatenate(data_dict['act_seq'], axis=0).astype(self._env_spec.names_to_dtypes['act'])

            split_indices = np.cumsum(data_dict['episode_sizes'])
            # remove the last chunk since it will be empty
            split_indices = np.delete(split_indices, -1)
            if self._split_indices.size > 0:
                self._split_indices = np.concatenate([self._split_indices, np.array([self._data_len]),
                                                      self._data_len + split_indices], axis=0)
            else:
                self._split_indices = split_indices

            self._num_episodes += len(data_dict['done'])
            self._data_len += np.sum(data_dict['episode_sizes'])
            logger.debug('Dataset length: {}'.format(self._data_len))

            for key in self._all_names:
                assert key in data_dict, f'{key} not in converted mat file'
                assert len(data_dict[key]) > 0
                # turn list into np array with the correct type
                local_dict[key] = np.concatenate(data_dict[key], axis=0).astype(self._env_spec.names_to_dtypes[key])
                assert local_dict[key].shape[1:] == self._env_spec.names_to_shapes[key], \
                    "Bad Data shape for {}: {}, requires {}" \
                        .format(key, local_dict[key].shape[1:], self._env_spec.names_to_shapes[key])
                # print(key, self._env_spec.names_to_shapes[key], local_dict[key].shape)
                assert local_dict[key].shape[0] == self._data_len, \
                    "Bad datalen for {}: {}, requires {}".format(key, local_dict[key].shape, self._data_len)

        return local_dict

    @abstract.overrides
    def get_output_stats(self):
        return AttrDict({
            'mu': self._mu_obs.copy(),
            'mu_delta': self._mu_delta_obs.copy(),
            'sigma': self._sigma_obs.copy(),
            'sigma_delta': self._sigma_delta_obs.copy(),
        })

    @abstract.overrides
    def get_batch(self, indices=None, torch_device=None, get_horizon_goals=False, get_action_seq=False, min_idx=0):
        # TODO fix this
        num_eps = len(self._datadict.done)  # number of episodes
        if indices is None:
            assert 0 <= min_idx < self._data_len
            batch = min(self._data_len - min_idx, self._batch_size)
            indices = np.random.choice(self._data_len - min_idx, batch, replace=False)
            indices += min_idx  # base index to consider in dataset

        # get current batch
        sampled_datadict = self._datadict.leaf_apply(lambda arr: arr[indices])

        inputs = AttrDict()
        outputs = AttrDict()
        goals = AttrDict()
        for key in self._env_spec.observation_names:
            inputs[key] = sampled_datadict[key]
        for key in self._env_spec.action_names:
            inputs[key] = sampled_datadict[key]

        for key in self._env_spec.output_observation_names:
            outputs[key] = sampled_datadict[key]

        outputs.done = sampled_datadict.done

        if torch_device is not None:
            for d in (inputs, outputs, goals):
                d.leaf_modify(lambda x: torch.from_numpy(x).to(torch_device))

        if get_action_seq:
            inputs['act_seq'] = torch.from_numpy(self._action_sequences[indices]).to(torch_device)

        if get_horizon_goals:
            for key in self._env_spec.goal_names:
                goals[key] = torch.from_numpy(sampled_datadict[key]).to(torch_device)

        if get_horizon_goals:
            return inputs, outputs, goals
        return inputs, outputs  # shape is (batch, horizon, name_dim...)

    # add an episode to the datadict
    @abstract.overrides
    def add_episode(self, obs, next_obs, goal, action, done):
        for oname in self._env_spec.observation_names:
            self._datadict[oname] = np.concatenate([self._datadict[oname], obs[oname].copy()], axis=0)
        for ooname in self._env_spec.output_observation_names:
            self._datadict[ooname] = np.concatenate([self._datadict[ooname], next_obs[ooname].copy()], axis=0)
        for aname in self._env_spec.action_names:
            self._datadict[aname] = np.concatenate([self._datadict[aname], action[aname].copy()], axis=0)
        for gname in self._env_spec.goal_names:
            self._datadict[gname] = np.concatenate([self._datadict[gname], goal[gname].copy()], axis=0)
        self._split_indices = np.append(self._split_indices, self._data_len)
        self._datadict.done = np.concatenate([self._datadict.done, done], axis=0)
        self._data_len += len(done)

    @abstract.overrides
    def add_sample(self, obs, next_obs, goal, action, done):
        # all elements should be (1, ..), done should be a bool
        assert isinstance(done, np.bool)

        for oname in self._env_spec.observation_names:
            self._datadict[oname] = np.concatenate([self._datadict[oname], obs[oname].copy()], axis=0)
        for ooname in self._env_spec.output_observation_names:
            self._datadict[ooname] = np.concatenate([self._datadict[ooname], next_obs[ooname].copy()], axis=0)
        for aname in self._env_spec.action_names:
            self._datadict[aname] = np.concatenate([self._datadict[aname], action[aname].copy()], axis=0)
        for gname in self._env_spec.goal_names:
            self._datadict[gname] = np.concatenate([self._datadict[gname], goal[gname].copy()], axis=0)

        if done:
            new_split = 0
            # search backwards for dones (safer than storing a separate state var)
            for j in range(len(self._datadict.done)):
                if self._datadict.done[-j-1]:
                    new_split = len(self._datadict.done) - j
                    break

            if new_split > 0:
                self._split_indices = np.append(self._split_indices, new_split)

        self._datadict.done = np.append(self._datadict.done, done)
        self._data_len += 1

    # called to reset to initial datadict state (clearing out episodes that were added)
    @abstract.overrides
    def reset(self):
        self._datadict = self._initial_datadict.copy()
        self._split_indices = self._initial_split_indices.copy()
        self._data_len = self._initial_data_len

    # don't call this function too often
    @abstract.overrides
    def save(self):
        save_dict = {}
        for key in self._all_names:
            save_dict[key] = np.concatenate(self._datadict[key])

        path = os.path.join(self._file_manager.exp_dir, self._output_file)
        np.savez_compressed(path, **save_dict)

    def __len__(self):
        return self._data_len

    @property
    def batch_size(self):
        return self._batch_size


if __name__ == '__main__':
    from mbrl.envs.env_spec import EnvSpec
    from dotmap import DotMap as d

    d = MatDataset(d(
        input_file='test.mat',
        output_file='delete',
        batch_size=5,
        horizon=9,  # effective batch of batch_size*(horizon+1)

    ), EnvSpec())
