import pytest

from padl import transform, group, IfTrain


CONST = 1


@transform
def x(y):
    return CONST + 1


def k(o):
    return o * 7


@transform
def y(y):
    return CONST + k(y)


@transform
def listcomp_a(y):
    return [x + CONST for x in y]


@transform
def listcomp_b(y):
    return [x + y for x in CONST]


@transform
def dictcomp_a(y):
    return {x: x + CONST for x in y}


@transform
def setcomp_a(y):
    return {x + CONST for x in y}


def maketransform():
    @transform
    def z(x):
        return x
    return z


@transform
class MyClassTransform:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def calculate(self, input_):
        return self.a * self.b + input_

    def __call__(self, input_):
        return self.calculate(input_) + y(input_) + self.c(input_)


def maketransformclass():
    @transform
    class MyClassTransform:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

        def calculate(self, input_):
            return self.a * self.b + input_

        def __call__(self, input_):
            return self.calculate(input_) + y(input_) + self.c(input_)

    return MyClassTransform


def makeclasstransform(a, b, c):
    @transform
    class MyClassTransform:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

        def calculate(self, input_):
            return self.a * self.b + input_

        def __call__(self, input_):
            return self.calculate(input_) + y(input_) + self.c(input_)

    return MyClassTransform(a, b, c)


def read_dump(name):
    with open(f'tests/material/dumps/{name}.txt') as f:
        return f.read()


def write_dump(dump, name):
    with open(f'tests/material/dumps/{name}.txt', 'w') as f:
        return f.write(dump)


def test_dump_a():
    assert x._pd_dumps() == read_dump('a')


def test_dump_b():
    assert y._pd_dumps() == read_dump('b')


def test_dump_class_a():
    t = MyClassTransform(1, 2, x)
    assert t._pd_dumps() == read_dump('class_a')


lambda_a = transform(lambda x: x)
lambda_b = transform(lambda x: y(x))


def test_lambda_a():
    assert lambda_a._pd_dumps() == read_dump('lambda_a')


def test_lambda_b():
    assert lambda_b._pd_dumps() == read_dump('lambda_b')


def test_nested_dump_a():
    assert maketransform()._pd_dumps() == read_dump('nested_a')


def test_nested_dump_c():
    t = makeclasstransform(1, 2, x)
    assert t._pd_dumps() == read_dump('nested_c')


c_a = x >> y >> x
c_b = x >> y >> x / x + c_a


def test_compound_dump_a():
    assert c_a._pd_dumps() == read_dump('compound_a')


def test_compound_dump_b():
    assert c_b._pd_dumps() == read_dump('compound_b')


g_a = x + group(y + x + x)


def test_grouped_dump_a():
    assert g_a._pd_dumps() == read_dump('grouped_a')


def test_if_train():
    assert IfTrain(x, y)._pd_dumps() == read_dump('iftrain')


def test_listcomp_a():
    assert listcomp_a._pd_dumps() == read_dump('list_comprehension_a')


def test_listcomp_b():
    assert listcomp_b._pd_dumps() == read_dump('list_comprehension_b')


def test_dictcomp_a():
    assert dictcomp_a._pd_dumps() == read_dump('dict_comprehension_a')


def test_setcomp_a():
    assert setcomp_a._pd_dumps() == read_dump('set_comprehension_a')


def test_with_raises():
    with open(__file__) as f:
        x = f.read()

    @transform
    def t(y):
        return x + y

    with pytest.raises(NotImplementedError):
        t._pd_dumps()
