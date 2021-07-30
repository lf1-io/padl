# pylint: disable=arguments-differ
"""Data set transforms"""
import json
import os
import pandas
import PIL.Image
from lf.variables import Variable
from lf.transforms.core import Transform


class DataSet(Transform):
    """
    :param: samples
    """
    def __init__(self, samples):

        super().__init__(
            samples=samples,
        )

    def __len__(self):
        return len(self.samples)

    def _get(self, x):
        return x

    def do(self, x):
        x = x % len(self.samples)
        x = self.samples[x]
        x = self._get(x)
        return x


class Images(DataSet):
    """
    :param: samples
    """
    def _get(self, x):
        return PIL.Image.open(x)


class ImageFolder(DataSet):
    """
    :param samples:
    :param folder:
    """
    def __init__(self, folder):
        super().__init__(
            samples=sorted(os.listdir(
                folder.value if isinstance(folder, Variable) else folder
            )),
        )
        self.folder = folder

    def _get(self, x):
        return PIL.Image.open(self.folder + '/' + x)


class JSON(DataSet):
    """
    JSON data set
    :param path_:
    """
    def __init__(self, path_):

        if isinstance(path_, Variable):
            with open(path_.value) as file1:
                self.records = json.load(file1)
        else:
            with open(path_) as file1:
                self.records = json.load(file1)

        if isinstance(self.records, dict):
            samples = list(self.records.keys())
        elif isinstance(self.records, list):
            samples = range(len(self.records))
        else:
            raise ValueError('JSON formatted not supported')

        super().__init__(
            samples=samples,
        )
        self.path_ = path_

    def _get(self, x):
        return self.records[x]


class CSV(DataSet):
    """CSV data set"""
    def __init__(self,
                 path_,
                 columns=None,
                 read_arguments=None):

        if isinstance(read_arguments, dict):
            if isinstance(path_, Variable):
                self.df = pandas.read_csv(path_.value, **read_arguments)
            else:
                self.df = pandas.read_csv(path_, **read_arguments)
        else:
            self.df = pandas.read_csv(path_)

        if columns is not None:
            self.df = self.df[columns]

        super().__init__(
            samples=range(len(self.df)),
        )
        self.path_ = path_
        self.columns = columns
        self.read_arguments = read_arguments

    def _get(self, x):
        return dict(self.df.iloc[x])


class Folds(Transform):
    def __init__(self, train_data, valid_data):
        super().__init__(train_data=train_data, valid_data=valid_data)

    def __len__(self):
        if self.stage == 'train':
            return len(self.train_data)
        return len(self.valid_data)

    def do(self, x):
        if self.stage == 'train':
            return self.train_data.do(x)
        return self.valid_data.do(x)
