from mbrl.envs.env_spec import EnvSpec
from mbrl.utils import abstract
from mbrl.utils.python_utils import AttrDict


class Dataset(abstract.BaseClass):

    def __init__(self, params, env_spec, file_manager):
        assert isinstance(params, AttrDict)
        assert isinstance(env_spec, EnvSpec)

        self._env_spec = env_spec
        self._file_manager = file_manager
        self._init_params_to_attrs(params)
        self._init_setup()

    @abstract.abstractmethod
    def _init_params_to_attrs(self, params):
        pass

    @abstract.abstractmethod
    def _init_setup(self):
        pass

    def get_output_stats(self):
        return AttrDict()

    @abstract.abstractmethod
    def get_batch(self):
        """
        Returns:
            inputs (AttrDict)
            outputs (AttrDict)
        """
        raise NotImplementedError
        inputs = AttrDict()
        outputs = AttrDict()
        return inputs, outputs

    @property
    def batch_size(self):
        return 0

    # dataset augmentation
    @abstract.abstractmethod
    def add_sample(self, obs, goal, action, done):
        pass

    @abstract.abstractmethod
    def add_episode(self, obs, goal, action, done):
        pass

    @abstract.abstractmethod
    def reset(self):
        pass

    # saving
    @abstract.abstractmethod
    def save(self):
        pass