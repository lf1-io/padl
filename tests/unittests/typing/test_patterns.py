from unittest import TestCase

from lf.typing.patterns import (
    Mvar, Mlist, Mtuple, Mismatch, convert, InstantiationCallback,
    NamedVars, uuid4, Counter, Ltuple, Llist, Madd, Mdict, Mmul
)


class TestPatterns(TestCase):
    def test_mvar(self):
        a = Mvar()
        b = Mvar(1)
        a @ b
        self.assertEqual(a, b)

    def test_mvar1(self):
        a = Mvar()
        b = Mvar()
        c = Mvar()
        a @ c
        self.assertEqual(a, c)
        b @ c
        self.assertEqual(b, c)
        self.assertEqual(b, a)

    def test_mlist(self):
        a = Mvar()
        b = Mvar()
        c = Mvar()
        d = Mvar()
        e = Mlist([a, b, b])
        f = Mlist([c, c, d])
        e @ f
        self.assertEqual(a, b)
        self.assertEqual(b, c)
        self.assertEqual(c, d)

    def test_mlist1(self):
        a = Mvar()
        b = Mvar()
        c = Mvar()
        d = Mvar()
        e = Mlist([a, b, b])
        f = Mvar()
        e @ f
        g = Mlist([c, c, d])
        f @ g
        self.assertEqual(a, b)
        self.assertEqual(b, c)
        self.assertEqual(c, d)

    def test_mlist_copy(self):
        a = Mvar()
        e = Mlist([a, a])
        f = e.copy({})
        self.assertEqual(f[0].id, f[1].id)

    def test_mtuple_copy(self):
        a = Mvar()
        e = Mtuple((a, a))
        f = e.copy({})
        self.assertEqual(f[0].id, f[1].id)

    def test_madd(self):
        a = Mvar()
        c = a + 1
        c @ 10
        self.assertEqual(c, 10)
        self.assertEqual(a, 9)

    def test_madd2(self):
        a, b = Mvar(), Mvar()
        with self.assertRaises(TypeError):
            a + b

    def test_madd3(self):
        a = Mvar()

        b = a + 5
        c = b + 5
        d = c + 5
        d @ 100
        self.assertEqual(d, 100)
        self.assertEqual(c, 95)
        self.assertEqual(b, 90)
        self.assertEqual(a, 85)

    def test_madd4(self):
        a = Mvar()
        b = a + 1
        c = a + 1
        b @ c

    def test_madd5(self):
        a = Mvar()

        b = a + 1
        c = a + 2
        with self.assertRaises(Mismatch):
            b @ c

    def test_convert(self):
        a = convert('?a', '1')
        b = convert('?a', '1')
        self.assertIs(a, b)

        self.assertIsInstance(convert(('?',)), Ltuple)
        self.assertIsInstance(convert([1, '?']), Llist)
        self.assertIsInstance(convert([1, '...']), Llist)
        self.assertIsInstance(convert({'a': '?'}), Mdict)
        self.assertIsInstance(convert('?a + 100'), Madd)
        self.assertIsInstance(convert('?a * 100'), Mmul)
        self.assertIsInstance(convert('100.'), float)
        self.assertIsInstance(convert('100.0'), float)
        self.assertIsInstance(convert('100'), int)

    def test_llist(self):
        a = convert([1, 2, 3])
        b = convert([1, 2, 3])
        a @ b
        a = convert([1, 2, '...a'])
        b = convert([1, 2, 3])
        a @ b
        self.assertEqual(a, b)
        a = convert([1, 2, '...a'])
        b = convert(['...3b'])
        a @ b
        self.assertEqual(a, b)
        a = convert(['...2a', 1, '...a'])
        b = convert(['...3b'])
        a @ b
        self.assertEqual(a, b)

    def test_llist_template(self):
        a = convert(['?', '?', '?'], template=1)
        self.assertEqual(a, convert([1, 1, 1]))

        a = convert(['?', '?', '?'], template=Counter())
        self.assertEqual(a, convert([0, 1, 2]))

        a = convert(['...a'], template=Counter())
        b = convert(['...3a'])
        a @ b
        self.assertEqual(a, convert([0, 1, 2]))
        a @ b  # at one point this caused a mismatch
        b @ a

    def test_llist_template_1(self):
        t = NamedVars()
        a = convert(['...a'], template=t)
        a @ convert(['...10a'])
        b = convert(['...10a'], template=t)
        for i in range(10):
            b[i] @ i
        self.assertEqual(a, convert(list(range(10))))

    def test_llist_template_2(self):
        uuid = str(uuid4())
        a = convert(['...a'], template=NamedVars(uuid=uuid))
        a @ convert(['...10a'])
        b = convert(['...10a'], template=NamedVars(uuid=uuid))
        for i in range(10):
            b[i] @ i
        self.assertEqual(a, convert(list(range(10))))

    def test_instantiation_callback(self):
        a = convert('?a')

        class TC(InstantiationCallback):
            def __call__(self, other):
                a @ 1

        b = convert('?b')
        b.instantiation_callbacks = [TC()]
        b @ 2
        self.assertEqual(a, 1)

    def test_instantiation_callback_1(self):
        # test if instantiation callbacks are copied to matched variables
        a = convert('?a')

        class TC(InstantiationCallback):
            def __call__(self, other):
                a @ 1

        b = convert('?b')
        c = convert('?c')
        b.instantiation_callbacks = [TC()]
        b @ c
        c @ 1
        self.assertEqual(a, 1)

    def test_freelist(self):
        a = convert(['...a'])
        b = convert('?')
        a @ b
        a = convert(['...a'])
        b = convert('?')
        b @ a
