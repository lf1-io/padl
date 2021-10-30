class SimpleDataset:
    """A simple dataset.

    Takes *samples* and applies *trans* to it.

    :param samples: An object implementing __getitem__ and __len__.
    :param trans: Preprocessing transform.
    :param exception: Exception to catch for (fall back to *default*).
    :param default: The default value to fall back to in case of exception.
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
        return self.trans(self.samples[item])

    def __len__(self):
        return len(self.samples)
