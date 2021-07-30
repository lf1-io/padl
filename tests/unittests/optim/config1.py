import random
import torch

from lf import transforms as lf


@lf.trans()
def loss(x, y):
    l_ = (x - y).pow(2).mean().sqrt()
    return l_


@lf.metrics.metric()
def precision(pred, y):
    return sum([pred[i] == y[i] for i in range(len(y))]) / len(y)


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.parameter * x


vd = lf.data.DataSet([random.random() for _ in range(10)])
gt = lf.data.DataSet([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
m = (
    lf.to_tensor
    >> lf.GPU(True)
    >> lf.TracedLayer(Dummy(), example=torch.randn(5), layer_name='test')
    >> lf.CPU(True)
    >> lf.x.item()
    >> lf.Lambda('lambda x: int(x > 0.5)')
)
target = (
    lf.to_tensor
    >> lf.GPU(True)
    >> lf.Identity()
)
td = (
    lf.data.DataSet([random.random() for _ in range(100)])
    + lf.data.DataSet([1 for _ in range(50)] + [0 for _ in range(50)])
)
tm = td >> (m.preprocess / target) >> loss
