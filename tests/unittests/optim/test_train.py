""" Test for trainer """
import pytest
import random
import tempfile
import torch
from torch.autograd import Variable
from torch.optim import Adam

from lf import transforms as lf
from lf.monitor.monitor import DefaultMonitor
from lf.optim.train import DefaultTrainer, SimpleTrainer


@pytest.fixture()
def tmp_dir():
    tmp_path = tempfile.TemporaryDirectory(dir='tests/material/')
    yield tmp_path.name
    tmp_path.cleanup()


@lf.trans()
def loss(x, y):
    l_ = (x - y).pow(2).mean().sqrt()
    return l_


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.parameter * x


validation_data = lf.data.DataSet([random.random() for _ in range(10)])
ground_truth = lf.data.DataSet([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
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

mon = DefaultMonitor(
    validation_data=validation_data,
    ground_truth=ground_truth,
    m=m,
    tm=tm,
    verbose=False,
    batch_size=5,
)


class TestDefaultTrainer:
    def test(self, tmp_dir):
        a_trainer = DefaultTrainer(
            tm=tm,
            experiment=tmp_dir,
            monitor=mon,
            num_workers=0,
            batch_size=5,
        )
        a_trainer.t.it = 50
        a_trainer.t.m.train()
        x = next(a_trainer.t.m.preprocess(range(100), **a_trainer.t.train_load))
        a_trainer.take_step(x)
        a_trainer.t.save()


class TrainerTest:

    model = None
    train_samples = None
    train_load = None
    trainer = None

    def test_trainer(self):
        self.model.train()
        data = next(self.model.preprocess(self.train_samples, **self.train_load))
        self.trainer.take_step(data)


@lf.trans()
def simple_loss(x):
    l_ = Variable(x.pow(torch.tensor(2)).mean().sqrt(), requires_grad=True)
    return l_


class TestSimpleTrainer(TrainerTest):
    @pytest.fixture(autouse=True)
    def init(self):
        self.train_samples = range(100)

        self.model = (
            lf.to_tensor
            >> lf.GPU(True)
            >> lf.TracedLayer(Dummy(), example=torch.randn(5), layer_name='test')
            >> lf.CPU(True)
                    )

        self.train_load = {
                            'batch_size': 5,
                            'verbose': False,
                            'flatten': False,
                            'shuffle': True,
                            'num_workers': 0,
                            'drop_last': True,
                        }

        optimizer = Adam([x for x in self.model.parameters() if x.requires_grad], lr=1e-3)

        self.trainer = SimpleTrainer(
                        model=self.model,
                        optimizer=optimizer,
                        loss=simple_loss,
                        train_samples=self.train_samples,
                        )
