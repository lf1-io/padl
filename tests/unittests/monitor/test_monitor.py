import random

from lf.monitor.monitor import DefaultMonitor
from lf.transforms.metrics import metric
from lf.transforms.data import DataSet
from lf.optim.train import TimeToSave
from lf import transforms as lf


@metric()
def precision(pred, y):
    return sum([pred[i] == y[i] for i in range(len(y))]) / len(y)


@lf.trans()
def loss(x, y):
    l_ = (x - y).pow(2).mean().sqrt()
    return l_


validation_data = DataSet([random.random() for _ in range(10)])
ground_truth = DataSet([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
m = (
    lf.to_tensor
    >> lf.GPU(True)
    >> lf.Identity()
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
    DataSet([random.random() for _ in range(100)])
    + DataSet([1 for _ in range(50)] + [0 for _ in range(50)])
)
tm = td >> (m.preprocess / target) >> loss


class TestDefaultMonitor:
    def test_metric(self):
        mon = DefaultMonitor(
            validation_data=validation_data,
            ground_truth=ground_truth,
            m=m,
            tm=tm,
            verbose=False,
            watch='precision',
            precision=precision,
            batch_size=5,
        )
        try:
            mon(0)
        except TimeToSave:
            pass

        past_results = mon.monitor.past_results
        assert 'precision' in past_results
        assert len(past_results['precision']) == 1

    def test_loss(self):
        mon = DefaultMonitor(
            validation_data=validation_data,
            ground_truth=ground_truth,
            m=m,
            tm=tm,
            verbose=False,
            batch_size=5,
        )
        try:
            mon(0)
        except TimeToSave:
            past_results = mon.monitor.past_results
            assert 'precision' not in past_results
