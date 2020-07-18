import abc as _abc
from abc import abstractmethod
import overrides as _overrides
from overrides import final, overrides

class BaseClass(_abc.ABC, _overrides.EnforceOverrides):
    pass

