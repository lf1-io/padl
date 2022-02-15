import pytest
import sys

from padl import transform, group, IfTrain, Batchify, Unbatchify, importdump, fulldump
import padl.dumptools.var2mod
from tests.material import transforms_in_module as tim


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
def indexset(x, i, j):
    x[i] = x[j]
    return x


@transform
class SelfAssign:
    def __init__(self, x):
        self.x = x
        self.y = self.x

    def __call__(self, x):
        return self.x + self.y + x


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


@transform
def recursive(x):
    if x == 0:
        return x
    return 1 + recursive(x - 1)


def maketransform():
    @transform
    def z(x):
        return x
    return z


@transform
def hangin_indent(a=1,
                  b=2):
    return a, b


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


def test_nested_dump_d():
    t = maketransformclass()(1, 2, x)
    assert t._pd_dumps() == read_dump('nested_d')


c_a = x >> y >> x
c_b = x >> y >> x / x + c_a
c_c = (x
    >> Batchify()
    >> y
    >> Unbatchify()
)


def test_pipeline_dump_a():
    assert c_a._pd_dumps() == read_dump('pipeline_a')


def test_pipeline_dump_b():
    assert c_b._pd_dumps() == read_dump('pipeline_b')


def test_pipeline_dump_c():
    assert c_c._pd_dumps() == read_dump('pipeline_c')


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


def test_recursive():
    assert recursive._pd_dumps() == read_dump('recursive')


def test_with_raises():
    with open(__file__) as f:
        x = f.read()

    @transform
    def t(y):
        return x + y

    with pytest.raises(NotImplementedError):
        t._pd_dumps()


def test_dumping_indexset():
    assert indexset._pd_dumps() == read_dump('indexset')


def test_dumping_selfassign():
    assert SelfAssign(1)._pd_dumps() == read_dump('selfassign')


def test_multiline_init():
    a = MyClassTransform(
        a=1,
        b=2,
        c=3
    )  # no further testing, this should just not fail


@transform
def f_using_dotimport(x):
    return padl.dumptools.var2mod.ast.parse(x)


def test_dotimport():
    dump = f_using_dotimport._pd_dumps()
    assert dump == read_dump('dotimport')


def test_dump_hanging_indent():
    dump = hangin_indent._pd_dumps()
    assert dump == read_dump('hanging_indent')


class TestOtherModule:
    def test_import_function(self):
        importdump(tim)
        dump = tim.function._pd_dumps()
        assert dump == read_dump('othermodule_import_function')

    def test_full_function(self):
        fulldump(tim)
        dump = tim.function._pd_dumps()
        assert dump == read_dump('othermodule_full_function')

    def test_import_class(self):
        importdump(tim)
        dump = tim.Class(1)._pd_dumps()
        assert dump == read_dump('othermodule_import_class')

    def test_full_class(self):
        fulldump(tim)
        dump = tim.Class(1)._pd_dumps()
        assert dump == read_dump('othermodule_full_class')

    def test_import_object(self):
        importdump(tim)
        dump = tim.obj._pd_dumps()
        assert dump == read_dump('othermodule_import_object')

    def test_full_object(self):
        fulldump(tim)
        dump = tim.obj._pd_dumps()
        assert dump == read_dump('othermodule_full_object')

    def test_import_pipeline(self):
        importdump(tim)
        dump = tim.pipeline._pd_dumps()
        assert dump == read_dump('othermodule_import_pipeline')

    def test_full_pipeline(self):
        fulldump(tim)
        dump = tim.pipeline._pd_dumps()
        assert dump == read_dump('othermodule_full_pipeline')

    def test_full_makefunction(self):
        fulldump(tim)
        dump = tim.makefunction()._pd_dumps()
        assert dump == read_dump('othermodule_full_makefunction')

    def test_full_makeclass(self):
        fulldump(tim)
        dump = tim.makeclass()(1, 2, 3)._pd_dumps()
        assert dump == read_dump('othermodule_full_makeclass')

    def test_full_makeclasstransform(self):
        fulldump(tim)
        dump = tim.makeclasstransform(1, 2, 3)._pd_dumps()
        assert dump == read_dump('othermodule_full_makeclasstransform')

    def test_full_makeclasstransform_with_constants(self):
        fulldump(tim)
        B = 2
        C = 3
        dump = tim.makeclasstransform(CONST, B, C)._pd_dumps()
        if sys.version_info[1] <= 8:
            assert dump == read_dump('othermodule_full_makeclasstransform_with_transforms')
        else:
            # python >= 3.9 creates a slightly different dump (semantically the same)
            assert dump == read_dump('othermodule_full_makeclasstransform_with_transforms38')

    def test_import_wrapped(self):
        importdump(tim)
        dump = tim.wrap_transform()._pd_dumps()
        assert dump == read_dump('othermodule_import_wrapped')

    def test_import_makelambda(self):
        importdump(tim)
        dump = tim.makelambda()._pd_dumps()
        assert dump == read_dump('othermodule_import_makelambda')

    def test_import_makefunction_squared(self):
        importdump(tim)
        dump = tim.makefunction_squared()._pd_dumps()
        assert dump == read_dump('othermodule_import_makefunction_squared')

    def test_import_makeclass_squared(self):
        importdump(tim)
        dump = tim.makeclass_squared()._pd_dumps()
        assert dump == read_dump('othermodule_import_makeclass_squared')
