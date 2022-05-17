from textwrap import dedent

from padl.dumptools import config_tools as ct
from padl.dumptools.ast_utils import Position


class TestParam:
    def test_input_is_output(self):
        assert 1 == ct.param('a', 1)


class TestParams:
    def test_output_is_dict(self):
        assert {'a': 1, 'b': 2, 'c': 3} == ct.params('a', a=1, b=2, c=3)


class TestExtractParamS:
    def test_multiple(self):
        s = dedent('''
            a = param('a', "1"); c = param('c', 1, 'hello', False)
            def x():
                b = param('bb', x)
        ''')
        p = ct.extract_param_s(s)
        assert p['a'] == {
            'name': "'a'",
            'val': '"1"',
            'description': 'None',
            'use_default': 'True',
            'position': Position(lineno=2, end_lineno=2, col_offset=15, end_col_offset=18)
        }
        assert p['c'] == {
            'name': "'c'",
            'val': '1',
            'description': "'hello'",
            'use_default': 'False',
            'position': Position(lineno=2, end_lineno=2, col_offset=36, end_col_offset=37)
        }
        assert p['bb'] == {
            'name': "'bb'",
            'val': 'x',
            'description': 'None',
            'use_default': 'True',
            'position': Position(lineno=4, end_lineno=4, col_offset=20, end_col_offset=21)
        }

    def test_defaults_are_filled_in(self):
        s = 'param("a", 1)'
        p = ct.extract_param_s(s)
        assert p['a'] == {
            'name': '"a"',
            'val': '1',
            'description': 'None',
            'use_default': 'True',
            'position': Position(lineno=1, end_lineno=1, col_offset=11, end_col_offset=12)
        }

    def test_defaults_can_be_overridden(self):
        s = 'param("a", 1, "bla", False)'
        p = ct.extract_param_s(s)
        assert p['a'] == {
            'name': '"a"',
            'val': '1',
            'description': '"bla"',
            'use_default': 'False',
            'position': Position(lineno=1, end_lineno=1, col_offset=11, end_col_offset=12)
        }

    def test_defaults_can_be_overridden_via_keywords(self):
        s = 'param(use_default=False, description="bla", name="a", val=1)'
        p = ct.extract_param_s(s)
        assert p['a'] == {
            'name': '"a"',
            'val': '1',
            'description': '"bla"',
            'use_default': 'False',
            'position': Position(lineno=1, end_lineno=1, col_offset=58, end_col_offset=59)
        }

    def test_correct_keys(self):
        s = 'param("a", 1)'
        p = ct.extract_param_s(s)
        assert set(p.keys()) == {'a'}
        assert set(p['a'].keys()) == {'name', 'val', 'description', 'use_default', 'position'}

    def test_complex_values_work(self):
        s = 'param("a", np.random.rand(100 ** 10))'
        p = ct.extract_param_s(s)
        assert p['a']['val'] == 'np.random.rand(100 ** 10)'

    def test_attribute_works(self):
        s = 'padl.param("a", 1)'
        p = ct.extract_param_s(s)
        assert p['a'] == {
            'name': '"a"',
            'val': '1',
            'description': 'None',
            'use_default': 'True',
            'position': Position(lineno=1, end_lineno=1, col_offset=16, end_col_offset=17)
        }

    def test_finds_nested(self):
        s = dedent('''
            class X:
                def f(self):
                    def o():
                        return [x for x in param('x', [1, 2, 3])]
        ''')
        p = ct.extract_param_s(s)
        assert 'x' in p


class TestExtractParamsS:
    def test_defaults_are_filled_in(self):
        s = 'params("a", x=1)'
        p = ct.extract_params_s(s)
        assert p['a'] == {
            'name': '"a"',
            'kwargs': {'x': '1'},
            'use_defaults': 'True',
            'allow_free': 'False',
            'positions': {'x': Position(lineno=1, end_lineno=1, col_offset=14, end_col_offset=15)},
            'end_position': Position(lineno=1, end_lineno=1, col_offset=15, end_col_offset=15)
        }

    def test_defaults_can_be_overridded(self):
        s = 'params("a", x=1, use_defaults=False, allow_free=True)'
        p = ct.extract_params_s(s)
        assert p['a'] == {
            'name': '"a"',
            'kwargs': {'x': '1'},
            'use_defaults': 'False',
            'allow_free': 'True',
            'positions': {'x': Position(lineno=1, end_lineno=1, col_offset=14, end_col_offset=15)},
            'end_position': Position(lineno=1, end_lineno=1, col_offset=15, end_col_offset=15)
        }

    def test_keywords_can_be_used_for_everything(self):
        s = 'params(use_defaults=False, name="a", x=1, allow_free=True)'
        p = ct.extract_params_s(s)
        assert p['a'] == {
            'name': '"a"',
            'kwargs': {'x': '1'},
            'use_defaults': 'False',
            'allow_free': 'True',
            'positions': {'x': Position(lineno=1, end_lineno=1, col_offset=39, end_col_offset=40)},
            'end_position': Position(lineno=1, end_lineno=1, col_offset=40, end_col_offset=40)
        }

    def test_correct_keys(self):
        s = 'params("a", x=1)'
        p = ct.extract_params_s(s)
        assert set(p.keys()) == {'a'}
        assert set(p['a'].keys()) == {'name', 'kwargs', 'allow_free', 'use_defaults', 'positions',
                                      'end_position'}

    def test_complex_values_work(self):
        s = 'params("a", x=np.random.rand(100 ** 10))'
        p = ct.extract_params_s(s)
        assert p['a']['kwargs']['x'] == 'np.random.rand(100 ** 10)'

    def test_attribute_works(self):
        s = 'padl.params("a", x=1)'
        p = ct.extract_params_s(s)
        assert 'a' in p

    def test_finds_nested(self):
        s = dedent('''
            class X:
                def f(self):
                    def o():
                        return [x for x in params('x', o=[1, 2, 3])['o']]
        ''')
        p = ct.extract_params_s(s)
        assert 'x' in p


class TestChangeParam:
    def test_simple(self):
        s = 'param("x", 1)'
        assert ct.change_param(s, 'x', '100') == 'param("x", 100)'

    def test_attribute_works(self):
        s = 'padl.param("x", 1)'
        assert ct.change_param(s, 'x', '100') == 'padl.param("x", 100)'

    def test_other_arguments_stay(self):
        s = 'param("x", 1, "this is a parameter", use_default=True)'
        assert ct.change_param(s, 'x', '100') == 'param("x", 100, "this is a parameter", use_default=True)'

    def test_weird_formatting_works(self):
        s = dedent('''
            param("x", 1,
                        "this is a parameter",
                    use_default=True

                )
        ''')
        target = dedent('''
            param("x", 100,
                        "this is a parameter",
                    use_default=True

                )
        ''')
        assert ct.change_param(s, 'x', '100') == target

    def test_nested_works(self):
        s = dedent('''
            def f():
                class X:
                    def __init__(self, y):
                        o = param("x", 1, "this is a parameter", use_default=True)
        ''')
        target = dedent('''
            def f():
                class X:
                    def __init__(self, y):
                        o = param("x", 100, "this is a parameter", use_default=True)
        ''')
        assert ct.change_param(s, 'x', '100') == target


class TestChangeParams:
    def test_simple(self):
        s = 'params("x", x=1)'
        assert ct.change_params(s, 'x', x='100') == 'params("x", x=100)'

    def test_attribute_works(self):
        s = 'padl.params("x", x=1)'
        assert ct.change_params(s, 'x', x='100') == 'padl.params("x", x=100)'

    def test_multiple(self):
        s = 'padl.params("x", x=1, y=2, z=3)'
        assert ct.change_params(s, 'x', x='100', z='1000') == 'padl.params("x", x=100, y=2, z=1000)'

    def test_other_arguments_stay(self):
        s = 'params("x", x=1, use_defaults=True, allow_free=False)'
        assert ct.change_params(s, 'x', x='100') == 'params("x", x=100, use_defaults=True, allow_free=False)'

    def test_weird_formatting_works(self):
        s = dedent('''
            params("x", x=1,
                    use_default=True

                )
        ''')
        target = dedent('''
            params("x", x=100,
                    use_default=True

                )
        ''')
        assert ct.change_params(s, 'x', x='100') == target

    def test_nested_works(self):
        s = dedent('''
            def f():
                class X:
                    def __init__(self, y):
                        o = params("x", x=1, use_default=True)
        ''')
        target = dedent('''
            def f():
                class X:
                    def __init__(self, y):
                        o = params("x", x=100, use_default=True)
        ''')
        assert ct.change_params(s, 'x', x='100') == target
