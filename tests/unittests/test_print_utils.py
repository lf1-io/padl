from padl.print_utils import FStr


class TestFStr:
    def test_1(self):

        FStr.use = False

        s1 = FStr('abc')
        s2 = FStr('def')

        print(s1 + s2)

        print(s1.sequence)