"""Variables for lf"""


class Variable:
    """
    :param name: name of variable
    :param value: value
    :param has_value: flag
    :param dont_store:
    """
    custom = False

    def __init__(self, name, value=None, has_value=False, dont_store=False):
        self.name = name
        self._value = value
        self._has_value = has_value or value is not None
        self.dont_store = dont_store

    @property
    def value(self):
        if self._has_value:
            return self._value
        else:
            raise AttributeError(
                f'Trying to retrieve value of empty Variable "{self.name}".')

    @value.setter
    def value(self, x):
        self._has_value = True
        self._value = x


class StateDict(Variable):
    ...

