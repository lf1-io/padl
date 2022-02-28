from padl.dumptools import sourceget


a = \
'''xxxx
01234567890
01234567890xxxxxxxx
01234567890xxx
01234567890
xxx
'''

a_r = \
'''xxxx
01234567890
01here567890xxxxxxxx
01234567890xxx
01234567890
xxx
'''

a_r1 = \
'''xxxx
01234567890
01he
re890xxxxxxxx
01234567890xxx
01234567890
xxx
'''

b = '0123456789'

b_r = '0123here6789'


class TestReplace:
    def test_a(self):
        assert sourceget.replace(a, 'here', 2, 2, 2, 5) == a_r

    def test_a1(self):
        assert sourceget.replace(a, 'he\nre', 2, 2, 2, 8) == a_r1

    def test_b(self):
        assert sourceget.replace(b, 'here', 0, 0, 4, 6) == b_r


class TestCut:
    def test_single_line(self):
        assert sourceget.cut(a, 2, 2, 1, 4) == '123'
