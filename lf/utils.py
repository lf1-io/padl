from lf import trans


class _This:
    """
    Transform factory for capturing attributes/ get-items
    """

    def __getitem__(self, item):
        return self.__getattr__('__getitem__')(item)

    def __getattr__(self, attr):
        @trans
        class T:
            """Dynamically generated transform for the "x" object.
            :param args: Arguments to pass to the input's method.
            :param kwargs: Keyword arguments to pass to the input's method.
            """

            def __init__(self, *args, **kwargs):
                self.__args = args
                self.__kwargs = kwargs
                super().__init__()

            def __call__(self, input_):
                return getattr(input_, attr)(*self.__args, **self.__kwargs)

        return T

    def __call__(self, fname):
        @trans
        class T:
            """
            """

            def __init__(self):
                super().__init__()

            def __call__(self, input_):
                return __builtins__[fname](input_)

        return T()


this = _This()