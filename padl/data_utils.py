import random


class Map:
    """ Map a transform to the data. """
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])


class RandomSubset:
    """ Randomly select a subset of the data """
    def __init__(self, data, n):
        self.data = data
        self.n = n
        self.indices = random.sample(range(len(data)), n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[self.indices[idx]]
