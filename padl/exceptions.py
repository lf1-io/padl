"""Exceptions."""


class WrongDeviceError(Exception):
    """Devices do not match for all transforms.

    :param mother_transform: Main transform that contains child transform.
    :param child_transform: Transform that is contained in mother transform.
    """

    def __init__(self, mother_transform, child_transform):
        super().__init__(f"{mother_transform._pd_shortrepr()} is in '{mother_transform.pd_device}',"
                         f' while {child_transform._pd_shortrepr()} is in '
                         f"'{child_transform.pd_device}'")
