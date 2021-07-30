import pytest
import torch

from lf.testing import TransformTest, assert_close
from lf.transforms import core as lf
from lf.typing.patterns import val


@lf.trans()
def plus_one(x):
    return x + 1


@lf.trans()
def append_one(x):
    return x + "one"


@lf.trans()
def times_two(x):
    return x * 2


@lf.trans()
def plus(x, y):
    return x + y


class DummyModel(torch.jit.ScriptModule):  # using this to test Layer
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.a = torch.jit.trace(
            torch.nn.Linear(n_input, n_hidden),
            torch.randn(2, n_input),
        )
        self.b = torch.jit.trace(
            torch.nn.Linear(n_hidden, n_output),
            torch.randn(2, n_hidden),
        )

    @torch.jit.script_method
    def forward(self, x):
        return self.b(self.a(x))


class TestPlusOne(TransformTest):
    transform = plus_one
    input_ = 1
    output = 2


class TestTimesTwo(TransformTest):
    transform = times_two
    inputs = [100]
    outputs = [200]


class TestLayer(TransformTest):
    @pytest.fixture(autouse=True)
    def init(self):
        model = DummyModel(10, 20, 30)
        self.input_ = torch.randn(10)
        model.eval()
        with torch.no_grad():
            self.output = model(self.input_.unsqueeze(0)).squeeze()

        self.transform = (
            lf.GPU(True)
            >> lf.Layer(model, layer_name='test_layer')
            >> lf.CPU(True)
        )


class TestTracedLayer(TransformTest):
    @pytest.fixture(autouse=True)
    def init(self):
        model = torch.nn.Linear(10, 20)
        self.input_ = torch.randn(10)
        model.eval()
        with torch.no_grad():
            self.output = model(self.input_.unsqueeze(0)).squeeze()

        tl = lf.TracedLayer(model, layer_name='test_layer', example=torch.randn(2, 10))
        self.transform = (
            lf.GPU(True)
            >> tl
            >> lf.CPU(True)
        )


class TestLambda(TransformTest):
    transform = lf.Lambda('lambda x: x[0] * x[1] + 1001')
    inputs = [(2, 5)]
    output = [1011]


class TestCompose(TransformTest):
    @pytest.fixture(autouse=True)
    def init(self):
        self.input_ = torch.Tensor([1, 1, 1])
        output = self.input_ + 1
        output *= 2
        dm = DummyModel(3, 3, 3)
        with torch.no_grad():
            output = dm(output.unsqueeze(0)).squeeze()
        output += 1
        output *= 2

        self.transform = (
            plus_one
            >> times_two
            >> lf.GPU(True)
            >> lf.Layer(dm, 'dm')
            >> lf.CPU(True)
            >> plus_one
            >> times_two
        )
        self.output = output

    def test_associative(self):
        a = (plus_one >> times_two) >> times_two
        b = plus_one >> (times_two >> times_two)
        in_ = 123
        assert a(in_) == b(in_)


class TestParallel(TransformTest):
    @pytest.fixture(autouse=True)
    def init(self):
        self.input_ = (torch.Tensor([1, 1, 1]), torch.Tensor([2, 2, 2]))
        dm = DummyModel(3, 3, 3)

        output0 = self.input_[0] + 1
        output0 = output0 * 2
        with torch.no_grad():
            output0 = dm(output0.unsqueeze(0)).squeeze()
        output0 = output0 + 1
        output0 = output0 * 2

        output1 = self.input_[1] * 2
        output1 = output1 + 1
        with torch.no_grad():
            output1 = dm(output1.unsqueeze(0)).squeeze()
        output1 = output1 * 2
        output1 = output1 + 1

        self.transform = (
            plus_one / times_two
            >> times_two / plus_one
            >> lf.GPU(True)
            >> lf.Layer(dm, 'dm') / lf.Layer(dm, 'dm1')
            >> lf.CPU(True)
            >> plus_one / times_two
            >> times_two / plus_one
        )
        self.output = (output0, output1)


class TestRollout(TransformTest):
    @pytest.fixture(autouse=True)
    def init(self):
        self.input_ = torch.Tensor([1, 1, 1])
        dm = DummyModel(3, 3, 3)

        output0 = self.input_ + 1
        output0 = output0 * 2
        with torch.no_grad():
            output0 = dm(output0.unsqueeze(0)).squeeze()

        output1 = self.input_ * 2
        output1 = output1 + 1
        with torch.no_grad():
            output1 = dm(output1.unsqueeze(0)).squeeze()

        output = output0 + output1
        output0 = output * 2
        output1 = output + 1

        self.transform = (
            plus_one + times_two
            >> times_two / plus_one
            >> lf.GPU(True)
            >> lf.Layer(dm, 'dm') / lf.Layer(dm, 'dm1')
            >> lf.CPU(True)
            >> plus
            >> times_two + plus_one
        )
        self.output = (output0, output1)


class TestMap(TransformTest):
    input_ = [1, 2, 3, 4, 5]
    output = [2, 3, 4, 5, 6]
    transform = ~ plus_one


class TestMap1(TransformTest):
    input_ = [1, 2, 3, 4, 5]
    output = [2, 4, 6, 8, 10]
    transform = ~ times_two


class TestBatchify(TransformTest):
    transform = lf.Batchify()
    input_ = (
        torch.Tensor(
            [1, 2, 3],
        ), (
            torch.Tensor(
                [4, 5, 6],
            ),
            torch.Tensor(
                [7, 8, 9],
            )
        )
    )
    output = (
        torch.Tensor(
            [[1, 2, 3]],
        ), (
            torch.Tensor(
                [[4, 5, 6]],
            ),
            torch.Tensor(
                [[7, 8, 9]],
            )
        )
    )

    def test_eval(self):
        pass  # does not apply

    def test_types_1(self, strict_types):
        @lf.trans(lf.Any(), lf.Tensor(shape=[100, 200]))
        def preprocess(x):
            return x

        batchify = lf.Batchify()
        composed = preprocess >> batchify
        assert list(composed.type.y['shape'][1:]) == [100, 200]
        assert composed.type.y['shape'][0].free

    def test_types_2(self, strict_types):
        @lf.trans(lf.Any(), lf.Tensor(shape=[100, 200]))
        def preprocess(x):
            return x

        batchify = lf.Batchify(dim=1)
        composed = preprocess >> batchify
        assert composed.type.y['shape'][0] == 100
        assert composed.type.y['shape'][2] == 200
        assert composed.type.y['shape'][1].free

    def test_types_3(self, strict_types):
        @lf.trans(lf.Tensor(shape=[100, 200, 300]), lf.Any())
        def forward(x):
            return x

        batchify = lf.Batchify()
        composed = batchify >> forward
        assert val(composed.type.x['shape'])[0] == 200
        assert val(composed.type.x['shape'])[1] == 300
        assert val(composed.type.x['shape']).len == 2

    def test_types_4(self, strict_types):
        @lf.trans(lf.Tensor(shape=[100, 200, 300]), lf.Any())
        def forward(x):
            return x

        batchify = lf.Batchify(dim=1)
        composed = batchify >> forward
        assert val(composed.type.x['shape'])[0] == 100
        assert val(composed.type.x['shape'])[1] == 300
        assert val(composed.type.x['shape']).len == 2

    def test_types_5(self, strict_types):
        @lf.trans(lf.Any(), (lf.Tensor(shape=[100, 200]), lf.Tensor(shape=[300, 400])))
        def preprocess(x):
            return x

        batchify = lf.Batchify()
        composed = preprocess >> batchify
        assert list(composed.type.y[0]['shape'][1:]) == [100, 200]
        assert list(composed.type.y[1]['shape'][1:]) == [300, 400]
        assert composed.type.y[0]['shape'][0].free
        assert composed.type.y[1]['shape'][0].free

    def test_types_5(self, strict_types):
        @lf.trans((lf.Tensor(shape=[100, 200, 300]), lf.Tensor(shape=[100, 400, 500])), lf.Any())
        def forward(x):
            return x

        batchify = lf.Batchify()
        composed = batchify >> forward
        assert list(val(composed.type.x[0]['shape'])) == [200, 300]
        assert list(val(composed.type.x[1]['shape'])) == [400, 500]
        assert val(composed.type.x[0]['shape']).len == 2
        assert val(composed.type.x[1]['shape']).len == 2


class TestUnbatchify(TransformTest):
    transform = lf.Unbatchify()
    output = (
        torch.Tensor(
            [1, 2, 3],
        ), (
            torch.Tensor(
                [4, 5, 6],
            ),
            torch.Tensor(
                [7, 8, 9],
            )
        )
    )
    input_ = (
        torch.Tensor(
            [[1, 2, 3]],
        ), (
            torch.Tensor(
                [[4, 5, 6]],
            ),
            torch.Tensor(
                [[7, 8, 9]],
            )
        )
    )

    def test_eval(self):
        pass  # does not apply


class TransformTests:
    def test_compose1(self):
        comp = plus_one >> times_two
        assert comp(1) == 4

    def test_equals(self):
        @lf.trans()
        def f(x):
            return x

        @lf.trans()
        def g(x):
            return x

        assert f == f
        assert f == f.clone()
        assert f >> g == f >> g
        assert f + g == f + g
        assert f / g == f / g
        assert f == g
        assert f.clone() == g
        assert f >> g == g >> f
        assert f + g == g + f
        assert f / g == g / f

    def test_forward(self):
        @lf.trans()
        def f(x):
            return x

        @lf.trans()
        def g(x):
            return x

        fg = f >> g
        fg1 = f >> lf.GPU() >> g
        fg2 = f >> g
        assert fg == fg2
        fg2.to_dict()
        assert fg1.trans == f

    def test_clone(self):
        assert plus_one == plus_one.clone()


class TestFlatten:
    def test_a(self):
        x = lf.flatten(((1, 2), 3, ((4,), 5)))
        assert x == (1, 2, 3, 4, 5)

    def test_b(self):
        x = lf.flatten(((1, 2, 3), (4, 5)))
        assert x == (1, 2, 3, 4, 5)


class TestChoose(TransformTest):
    transform = ~ lf.Choose(plus_one, times_two, lf.Lambda('lambda x: x > 5'))
    input_ = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    output = [2, 4, 6, 8, 10, 7, 8, 9, 10]

    def test_eval(self):
        pass


class TestIf(TransformTest):
    transform = ~ lf.If(plus_one, lf.Lambda('lambda x: x > 5'), times_two)
    input_ = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    output = [2, 4, 6, 8, 10, 7, 8, 9, 10]


class TestIfInStage:
    def test_eval(self):
        t = lf.IfInStage(plus_one, 'eval', times_two)
        t.eval()
        assert_close(list(t([1, 2, 3], num_workers=0)), [2, 3, 4])
        t.infer()
        assert_close(t(5), 10)
        t.train()
        assert_close(list(t([1, 2, 3], num_workers=0)), [2, 4, 6])

    def test_infer(self):
        t = lf.IfInStage(plus_one, 'infer', times_two)
        t.eval()
        assert_close(list(t([1, 2, 3], num_workers=0)), [2, 4, 6])
        t.infer()
        assert_close(t(5), 6)
        t.train()
        assert_close(list(t([1, 2, 3], num_workers=0)), [2, 4, 6])

    def test_train(self):
        t = lf.IfInStage(plus_one, 'train', times_two)
        t.eval()
        assert_close(list(t([1, 2, 3], num_workers=0)), [2, 4, 6])
        t.infer()
        assert_close(t(5), 10)
        t.train()
        assert_close(list(t([1, 2, 3], num_workers=0)), [2, 3, 4])


class TestIdentity(TransformTest):
    transform = lf.Identity()
    input_ = 123
    output = 123


class TestGet(TransformTest):
    transform = lf.Get('x', default=999)
    input_ = {'a': 1, 'b': 2, 'x': 1000, 'y': 123}
    output = 1000


class TestGet1(TransformTest):
    transform = lf.Get('z', default=999)
    input_ = {'a': 1, 'b': 2, 'x': 1000, 'y': 123}
    output = 999


class TestTupleGet(TransformTest):
    transform = lf.TupleGet(2)
    input_ = (0, 1, 2, 3)
    output = 2


class TestTry(TransformTest):
    transform = lf.Try(plus_one, append_one, (TypeError,))
    input_ = 1
    output = 2

    def test_eval(self):
        pass


class TestTry1(TransformTest):
    transform = lf.Try(plus_one, append_one, (TypeError,))
    input_ = 'one'
    output = 'oneone'

    def test_eval(self):
        pass


class TestTry2(TransformTest):
    transform = ~ lf.Try(plus_one, append_one, (TypeError,))
    input_ = [1, '1', 2, 'one']
    output = [2, '1one', 3, 'oneone']

    def test_eval(self):
        pass


class TestForceStage(TransformTest):
    transform = lf.ForceStage(lf.IfInStage(plus_one, 'eval', times_two), 'eval')
    input_ = 1
    output = 2


class TestForceStage1(TransformTest):
    transform = ~ lf.ForceStage(lf.IfInStage(plus_one, 'eval', times_two), 'eval')
    input_ = [1, 2, 3]
    output = [2, 3, 4]


class TestForceStage2(TransformTest):
    transform = ~ lf.ForceStage(lf.IfInStage(plus_one, 'eval', times_two), 'train')
    input_ = [1, 2, 3]
    output = [2, 4, 6]


class TestX(TransformTest):
    transform = lf.x[0]
    input_ = [1, 2, 3]
    output = 1


class TestX1(TransformTest):
    transform = lf.x['b']
    input_ = {'a': 1, 'b': 2, 'c': 3}
    output = 2


class TestX2(TransformTest):
    transform = lf.x.is_integer()
    input_ = 2.0
    output = True


class TestValue(TransformTest):
    transform = lf.Value(123)
    input_ = 'haha'
    output = 123


class TestValue1(TransformTest):
    transform = lf.Value(123)
    input_ = None
    output = 123


class TestMismatch:
    @pytest.fixture(autouse=True)
    def init(self):
        self.prev = lf.settings['strict_types']
        yield
        lf.settings['strict_types'] = self.prev

    def test_mismatch(self):
        a = lf.Transform(out_type=lf.Tensor())
        b = lf.Transform(in_type=lf.Float())
        lf.settings['strict_types'] = 'strict'
        with pytest.raises(lf.TypeMismatch):
            a >> b

    def test_wrong_setting(self):
        a = lf.Transform(out_type=lf.Tensor())
        b = lf.Transform(in_type=lf.Float())
        lf.settings['strict_types'] = 'huaa'
        with pytest.raises(AssertionError):
            a >> b


class TestBase:
    def test_wrong_param(self):
        for p in ['trans', 'postprocess', 'forward', 'to_dict', 'from_dict']:
            with pytest.raises(ValueError):
                lf.Transform(**{p: 1})

    def test_set_type(self):
        t = lf.Transform()
        assert t.type.x.type.rval.name == "Any"
        assert t.type.y.type.rval.name == "Any"
        t.set_type(out_type=lf.String())
        t.set_type(in_type=lf.Float())
        assert t.type.x.type.rval.name == "Float"
        assert t.type.y.type.rval.name == "String"
