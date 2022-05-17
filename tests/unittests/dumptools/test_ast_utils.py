import ast
from textwrap import dedent

from padl.dumptools import ast_utils
from padl.dumptools.sourceget import cut


class TestPosition:
    def test_init(self):
        pos = ast_utils.Position(1, 2, 3, 4)
        assert pos.lineno == 1
        assert pos.end_lineno == 2
        assert pos.col_offset == 3
        assert pos.end_col_offset == 4

    def test_iter(self):
        pos = ast_utils.Position(1, 2, 3, 4)
        assert list(pos) == [1, 2, 3, 4]


class TestCachedParse:
    def test_same_as_ast_parse(self):
        source = dedent('''
            a = 1
            b = 2

            class X:
                def haha(self):
                    return [x for x in range(123)]
        ''')
        assert ast_utils.dump(ast_utils.cached_parse(source)) == \
            ast_utils.dump(ast_utils.ast.parse(source))

    def test_cache(self):
        source = dedent('''
            a = 1
            b = 2

            class X:
                def haha(self):
                    return [x for x in range(123)]
        ''')
        assert ast_utils.cached_parse(source) is ast_utils.cached_parse(source)


class TestGetSourceSegment:
    def test_get_function_def(self):
        function = dedent('''\
            def haha(self):
                return [x for x in range(123)]
        ''')
        source = (
            'a = 1\n'
            'b = 2\n'
            + function +
            'c = 3'
        )
        parsed = ast.parse(source).body[2]
        assert ast_utils.get_source_segment(source, parsed).strip() == function.strip()

    def test_get_assign(self):
        assign = dedent('''\
            d = 123
        ''')
        source = (
            'a = 1\n'
            'b = 2\n'
            + assign +
            'c = 3'
        )
        parsed = ast.parse(source).body[2]
        assert ast_utils.get_source_segment(source, parsed).strip() == assign.strip()

    def test_get_nested_equal(self):
        equal = 'x == 123'
        source = (
            'a = 1\n'
            'b = 2\n'
            f'o = [x for x in bla if {equal}]\n'
            'c = 3'
        )
        parsed = ast.parse(source).body[2].value.generators[0].ifs[0]

        assert ast_utils.get_source_segment(source, parsed).strip() == equal.strip()


class TestGetPosition:
    def test_get_function_def(self):
        function = dedent('''\
            def haha(self):
                return [x for x in range(123)]
        ''')
        source = (
            'a = 1\n'
            'b = 2\n'
            + function +
            'c = 3'
        )
        parsed = ast.parse(source).body[2]
        assert cut(source, *ast_utils.get_position(source, parsed), one_indexed=True) == \
            function.strip()

    def test_get_assign(self):
        assign = dedent('''\
            d = 123
        ''')
        source = (
            'a = 1\n'
            'b = 2\n'
            + assign +
            'c = 3'
        )
        parsed = ast.parse(source).body[2]
        assert cut(source, *ast_utils.get_position(source, parsed), one_indexed=True) == \
            assign.strip()

    def test_get_nested_equal(self):
        equal = 'x == 123'
        source = (
            'a = 1\n'
            'b = 2\n'
            f'o = [x for x in bla if {equal}]\n'
            'c = 3'
        )
        parsed = ast.parse(source).body[2].value.generators[0].ifs[0]

        assert cut(source, *ast_utils.get_position(source, parsed), one_indexed=True) == \
            equal.strip()


class TestSpanToPos:
    def test_single_line(self):
        text = '0123456789'
        assert ast_utils.span_to_pos((1, 3), text) == \
            ast_utils.Position(1, 1, 1, 3)

    def test_single_line_in_multiline_text(self):
        text = ('0123456789\n'
                '0123456789\n'
                '0123456789')
        assert ast_utils.span_to_pos((12, 14), text) == \
            ast_utils.Position(2, 2, 1, 3)

    def test_multiline(self):
        text = ('0123456789\n'
                '0123456789\n'
                '0123456789')
        assert ast_utils.span_to_pos((12, 25), text) == \
            ast_utils.Position(2, 3, 1, 3)
