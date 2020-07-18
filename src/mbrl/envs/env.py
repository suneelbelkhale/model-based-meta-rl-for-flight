from mbrl.utils import abstract


class Env(abstract.BaseClass):

    def __init__(self, params, env_spec):
        self._env_spec = env_spec

    @abstract.abstractmethod
    def step(self, action, **kwargs):
        raise NotImplementedError
        return obs, goal, done

    @abstract.abstractmethod
    def reset(self):
        raise NotImplementedError
        return obs, goal
