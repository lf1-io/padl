"""Exceptions"""


class WrongDeviceError(Exception):
    """Devices do not match for all transforms

    :param mother_transform: main transform that contains child transform
    :param child_transform: transform that is contained in mother transform
    """
    def __init__(self, mother_transform, child_transform):
        self.mother_name = mother_transform.td_name if mother_transform.td_name is not None else mother_transform._td_shortname()
        self.child_name = child_transform.td_name if child_transform.td_name is not None else child_transform._td_shortname()

        super().__init__(f"{self.mother_name} is in '{mother_transform.td_device}', while {self.child_name} is in '{child_transform.td_device}'")
