import numpy as np
import torch
import os

from mbrl.datasets.dataset import Dataset
from mbrl.experiments import logger
from mbrl.utils import abstract, file_utils, np_utils
from mbrl.utils.python_utils import AttrDict


class NpDataset(Dataset):

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        self._input_file = params.input_file  # None if we are starting a new file
        self._output_file = params.output_file
        self._batch_size = params.batch_size  # number of episodes per batch
        self._horizon = params.horizon  # obs sequence length per episode per batch
        self._save_every_n_steps = params.save_every_n_steps

        self._all_names = self._env_spec.names + ['done', 'rollout_timestep']
        self._last_save = 0
        self._data_len = 0

    @abstract.overrides
    def _init_setup(self):
        self._datadict, self._split_indices = self._load_np()

    def _load_np(self):
        local_dict = AttrDict()

        if self._input_file is None:
            for key in self._all_names:
                local_dict[key] = []
            split_indices = np.array([])
        else:
            logger.debug('Loading ' + self._input_file)
            datadict = np.load(self._input_file, mmap_mode="r", allow_pickle=True)

            self._data_len = len(datadict['done'])
            split_indices = np.where(datadict['done'])[0] + 1  # one after each episode ends
            # remove the last chunk since it will be empty
            if 0 < self._data_len == split_indices[-1]:
                split_indices = np.delete(split_indices, -1)

            for key in self._all_names:
                assert key in datadict, f'{key} not in np file'
                assert len(datadict[key]) == self._data_len
                # split by episode
                local_dict[key] = np.split(datadict[key], split_indices)

            logger.debug('Dataset length: {}'.format(self._data_len))

        return local_dict, split_indices

    def hor_chunk(self, array):
        rg = max(len(array) - self._horizon - 1, 1)
        idx = np.random.choice(rg)
        return array[idx:idx+self._horizon+1]

    @abstract.overrides
    def get_batch(self, indices=None, torch_device=None):
        # TODO fix this
        assert indices is None
        num_eps = len(self._datadict.done)  # number of episodes
        indices = np.random.choice(num_eps, self._batch_size, replace=False)

        sampled_datadict = self._datadict.leaf_apply(
            lambda list_of_arr: np.stack([self.hor_chunk(list_of_arr[i]) for i in indices]))

        inputs = AttrDict()
        outputs = AttrDict()
        for key in self._env_spec.observation_names:
            inputs[key] = sampled_datadict[key]
        for key in self._env_spec.action_names:
            inputs[key] = sampled_datadict[key]

        for key in self._env_spec.output_observation_names:
            outputs[key] = sampled_datadict[key]

        outputs.done = sampled_datadict.done.astype(bool)

        if torch_device is not None:
            for d in (inputs, outputs):
                d.leaf_modify(lambda x: torch.from_numpy(x).to(torch_device))

        return inputs, outputs  # shape is (batch, horizon, name_dim...)

    def add_episode(self, obs, goal, action, done):
        for oname in self._env_spec.observation_names:
            self._datadict[oname].append(obs[oname].copy())
        for aname in self._env_spec.action_names:
            self._datadict[aname].append(action[aname].copy())
        for gname in self._env_spec.goal_names:
            self._datadict[gname].append(goal[gname].copy())
        self._split_indices = np.append(self._split_indices, self._data_len)
        self._datadict.done.append(done)
        self._datadict.rollout_timestep.append(np.arange(len(done)))
        self._data_len += len(done)

        if self._data_len - self._last_save >= self._save_every_n_steps:
            print("SAVING:", self._data_len)
            self.save()
            self._last_save = self._data_len

    # don't call this function too often
    def save(self):
        save_dict = {}
        for key in self._all_names:
            save_dict[key] = np.concatenate(self._datadict[key])

        path = os.path.join(self._file_manager.exp_dir, self._output_file)
        np.savez_compressed(path, **save_dict)

    def __len__(self):
        return self._data_len


if __name__ == '__main__':
    from mbrl.envs.env_spec import EnvSpec
    from dotmap import DotMap as d
    d = NpDataset(d(
            input_file=None,
            output_file='delete',
            batch_size=5,
            horizon=9,  # effective batch of batch_size*(horizon+1)

        ), EnvSpec())

