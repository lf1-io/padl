"""Data utilities. """


class SimpleDataset:
    """A simple dataset.

    Takes *samples* and applies *transform* to it.

    :param samples: An object implementing __getitem__ and __len__.
    :param transform: Preprocessing transform.
    :param exception: Exception to catch for (fall back to *default*).
    :param default: The default value to fall back to in case of exception.
    """

    def __init__(self, samples, transform, exception=None, default=None):
        self.samples = samples
        self.transform = transform
        self.exception = exception
        self.default = default

    def __getitem__(self, item):
        if self.exception:
            try:
                return self.transform(self.samples[item])
            except self.exception:
                return self.default
        return self.transform(self.samples[item])

    def __len__(self):
        return len(self.samples)
