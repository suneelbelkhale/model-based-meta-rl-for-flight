from mbrl.utils import abstract


class Policy(abstract.BaseClass):

    def __init__(self, params, env_spec):
        self._env_spec = env_spec
        self._init_params_to_attrs(params)
        self._init_setup()

    @abstract.abstractmethod
    def _init_params_to_attrs(self, params):
        pass

    @abstract.abstractmethod
    def _init_setup(self):
        pass

    @abstract.abstractmethod
    def warm_start(self, model, observation, goal):
        raise NotImplementedError

    @abstract.abstractmethod
    def get_action(self, model, observation, goal, batch=False):
        """
        Args:
            model (Model):
            observation (AttrDict):
            goal (AttrDict):
            batch (bool):

        Returns:
            AttrDict
        """
        raise NotImplementedError

    def _set_fn(self, name, func, ftype):
        if func is None:
            return

        def _internal_setter(fn: ftype):
            self.__setattr__(name, fn)
        _internal_setter(func)