from lf.optim.loader import Loader


class TestLoader:
    def test_1(self):
        l = Loader('tests/unittests/optim', config='config1')
        t = l(batch_size=10)
        with t.t.m.set_stage('train'):
            l = next(t.t.m(range(100), num_workers=0, verbose=False, flatten=False))
        assert len(l.shape) == 0

    def test_2(self):
        l = Loader('tests/unittests/optim', config='config2')
        t = l(batch_size=10)
        with t.t.m.set_stage('train'):
            l = next(t.t.m(range(100), num_workers=0, verbose=False, flatten=False))
        assert len(l.shape) == 0
