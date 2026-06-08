import abc
from .serialization import Serializable

class NamedObject(Serializable, metaclass=abc.ABCMeta):
    _name = '__noname'

    @property
    def name(self):
        return self._name

    def has_name(self):
        return self._name != '__noname'


class LazyEditable:
    def __getattr__(self, name):
        if name == '_dirty':
            setattr(self, '_dirty', False)
        return False

    def is_dirty(self):
        return self._dirty

    def set_dirty(self):
        self._dirty = True

    def clear_dirty(self):
        self._dirty = False
