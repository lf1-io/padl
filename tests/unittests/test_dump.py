from lf import trans, group


CONST = 1


@trans
def x(y):
    return CONST + 1


def k(o):
    return o * 7


@trans
def y(y):
    return CONST + k(y)


def maketransform():
    @trans
    def z(x):
        return x
    return z


@trans
class MyClassTransform:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def calculate(self, input_):
        return self.a * self.b + input_

    def __call__(self, input_):
        return self.calculate(input_) + y(input_) + self.c(input_)


def read_dump(name):
    with open(f'tests/material/dumps/{name}.txt') as f:
        return f.read()


def write_dump(dump, name):
    with open(f'tests/material/dumps/{name}.txt', 'w') as f:
        return f.write(dump)


def test_dump_a():
    assert x.lf_dumps() == read_dump('a')


def test_dump_b():
    y.lf_dumps() == read_dump('b')


def test_dump_class_a():
    MyClassTransform(1, 2, x).lf_dumps() == read_dump('class_a')


lambda_a = trans(lambda x: x)
lambda_b = trans(lambda x: y(x))


def test_lambda_a():
    assert lambda_a.lf_dumps() == read_dump('lambda_a')


def test_lambda_b():
    assert lambda_b.lf_dumps() == read_dump('lambda_b')


def test_nested_dump_a():
    maketransform().lf_dumps() == read_dump('nested_a')


c_a = x >> y >> x
c_b = x >> y >> x / x + c_a


def test_compound_dump_a():
    assert c_a.lf_dumps() == read_dump('compound_a')


def test_compound_dump_b():
    assert c_b.lf_dumps() == read_dump('compound_b')


g_a = x + group(y + x + x)


def test_grouped_dump_a():
    assert g_a.lf_dumps() == read_dump('grouped_a')
