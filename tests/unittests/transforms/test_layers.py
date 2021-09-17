import torch

from lf.transforms.layers import Layer


class _SimpleLayer(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.jit.trace(
            torch.nn.Linear(16, 32),
            torch.randn(1, 16)
        )

    def forward(self, x):
        return self.linear(x)


class _SimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)

    def forward(self, x):
        return self.linear(x)


def test_get_script_layer():
    layer = Layer(_SimpleLayer(), 'test')
    input_ = torch.randn(2, 16)
    test1 = layer.do(input_)
    d, v = layer.to_dict()
    recon = Layer.from_dict(d['kwargs'], v)
    test2 = recon(input_)
    diff = (test1 - test2).pow(2).sum(1).sqrt().mean(0)
    assert diff.item() == 0


def test_get_module_layer():
    layer = Layer(_SimpleModule(), 'test')
    input_ = torch.randn(2, 16)
    test1 = layer.do(input_)
    d, v = layer.to_dict()
    recon = Layer.from_dict(d['kwargs'], v)
    test2 = recon(input_)
    diff = (test1 - test2).pow(2).sum(1).sqrt().mean(0)
    assert diff.item() == 0


def test_get_traced_layer():
    layer = Layer(
        _SimpleModule(),
        'test',
        example=torch.randn(1, 16),
    )
    input_ = torch.randn(2, 16)
    test1 = layer.do(input_)
    d, v = layer.to_dict()
    recon = Layer.from_dict(d['kwargs'], v)
    test2 = recon(input_)
    diff = (test1 - test2).pow(2).sum(1).sqrt().mean(0)
    assert diff.item() == 0
