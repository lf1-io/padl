"""Exceptions"""


class NoneTypeError(Exception):
    """
    NoneTypeError

    :param value: Error value
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


exceptions = {'NoneTypeError': NoneTypeError}
