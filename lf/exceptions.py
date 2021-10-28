"""Exceptions"""


class WrongDeviceError(Exception):
    """Devices do not match for all transforms

    :param mother_transform: main transform that contains child transform
    :param child_transform: transform that is contained in mother transform
    """
    def __init__(self, mother_transform, child_transform):
<<<<<<< HEAD
        self.mother_name = mother_transform.lf_name if mother_transform.lf_name is not None else mother_transform._lf_shortname()
        self.child_name = child_transform.lf_name if child_transform.lf_name is not None else child_transform._lf_shortname()
=======
        self.mother_name = mother_transform.lf_name if mother_transform.lf_name is not None else mother_transform.lf_shortname()
        self.child_name = child_transform.lf_name if child_transform.lf_name is not None else child_transform.lf_shortname()
>>>>>>> c6e8ce8 (tests)

        super().__init__(f"{self.mother_name} is in '{mother_transform.lf_device}', while {self.child_name} is in '{child_transform.lf_device}'")
