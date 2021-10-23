class SimpleIterator:  # TODO: find a different name (it's not an iterator)
    """
    :param samples: object implementing __getitem__ and __len__
    :param trans: instance of lf.lf.Transform
    :param exception: exception to catch for default
    :param default: default value
    """
    def __init__(self, samples, trans, exception=None, default=None):
        self.samples = samples
        self.trans = trans
        self.exception = exception
        self.default = default

    def __getitem__(self, item):
        if self.exception:
            try:
                return self.trans(self.samples[item])
            except self.exception:
                return self.default
        else:
            return self.trans(self.samples[item])

    def __len__(self):
        return len(self.samples)
