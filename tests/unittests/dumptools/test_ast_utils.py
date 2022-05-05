from padl.dumptools import ast_utils


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
