"""Exceptions"""


class WrongDeviceError(Exception):
    """Devices do not match for all transforms

    :param mother_transform: main transform that contains child transform
    :param child_transform: transform that is contained in mother transform
    """
    def __init__(self, mother_transform, child_transform):
        self.mother_name = mother_transform.pd_name if mother_transform.pd_name is not None else mother_transform._pd_shortname()
        self.child_name = child_transform.pd_name if child_transform.pd_name is not None else child_transform._pd_shortname()

        super().__init__(f"{self.mother_name} is in '{mother_transform.pd_device}', while {self.child_name} is in '{child_transform.pd_device}'")
