"""Typing patterns"""
from copy import copy as copycopy
from contextlib import contextmanager
import itertools
from uuid import uuid4
import re
import weakref

DEBUG = [False]

settings = {
    'debug': False
}


class Mismatch(Exception):
    """ Exception that is raised when two *Matchable*s cannot be matched."""
    ...


def val(value):
    """Get the value of x (if x is a MVar, else just x)."""
    try:
        return value.rval
    except AttributeError:
        return value


def free(value):
    """Check if x is free."""
    try:
        return value.free
    except AttributeError:
        return False


def precedence(value):
    """Return member variable precedence"""
    try:
        return value.precedence
    except AttributeError:
        return -1


def r(value):
    """Return member variable r"""
    try:
        return value.r
    except AttributeError:
        return value


def repr_(value):
    """Return string representation of value"""
    if isinstance(value, dict):
        out = ','.join([f"'{k}':" + repr_(v) for k, v in value.items()])
        return '{' + out + '}'
    a_val = val(value)
    if a_val is None:
        a_val = value
    try:
        return a_val.repr_()
    except AttributeError:
        try:
            return a_val.repr()
        except AttributeError:
            return repr(a_val)


def copy(value, mappings, uuid=None):
    """Copy value"""
    try:
        return value.copy(mappings, uuid)
    except (TypeError, AttributeError):
        return copycopy(value)


class Modifiable:
    """Modifiable base class"""
    def __add__(self, other):
        return Madd(self, other)

    def __sub__(self, other):
        return Msub(self, other)

    def __mul__(self, other):
        return Mmul(self, other)

    def __truediv__(self, other):
        return Mdiv(self, other)


class Matchable:
    """ Base class for things that can be matched using the *@* operator.
    For instance variables (-> Mvar), variable length sequences (-> Lsequence) or Types.

    The *@* operator is the *match* operator. Matching two things means to check if they match and
    if that's the case, match them. What does or does not match is defined by the subclasses.

    For instance, variables match if their values are equal or if at least one of them is free.
    If they are free, they match anything.

    >>> a = Mvar()
    >>> b = Mvar()
    >>> a
    ?a
    >>> b
    ?b
    >>> a @ b
    ?a
    >>> b  # a and b are now the same
    ?a
    >>> b @ 1
    1
    >>> a  # -> a and b are the same
    1
    >>> a @ 2
    -> Mismatch

    Matching sequences works by recusively matching all elements:

    >>> a = Mvar()
    >>> b = Mvar()
    >>> c = Mvar()
    >>> d = Llist.fromlist([a, b, c])
    >>> e = Llist.fromlist([a, 2, 1])
    >>> d
    [?a, ?b, ?c]
    >>> d @ e
    [?a, 2, 1]
    >>> b
    2
    >>> c
    1

    *Matchable* has a utility-method *convert* that can be used to create matchables.
    How the conversion works is implemented in the subclasses via the methods
    *_can_be_converted* and *_convert*. *_can_be_converted* checks if something can be converted
    to an instance of the respective *Matchable*subclass. *_convert* does the converting.
    In addition, a subclass can have the property *convert_precedence*, which determines the order
    of conversion if two or more subclasses can potentially convert the same thing.
    """
    convert_precedence = 0
    env = {}
    context = None

    def __matmul__(self, other):
        if precedence(r(other)) >= precedence(r(self)):
            if hasattr(r(other), 'match'):
                return r(other).match(r(self))

            if r(other) == r(self):
                return self
            raise Mismatch
        return r(self).match(r(other))

    def __rmatmul__(self, other):
        if precedence(r(other)) >= precedence(r(self)):
            if hasattr(r(other), 'match'):
                return r(other).match(r(self))

            if r(other) == r(self):
                return self
            raise Mismatch
        return r(self).match(r(other))

    class set_env:
        """Set environment"""
        def __init__(self, env):
            self.env = env

        def __enter__(self):
            Matchable.env[self.env] = True

        def __exit__(self, exc_type, exc_val, exc_tb):
            Matchable.env[self.env] = False

    @classmethod
    def in_env(cls, env):
        """Return environment"""
        return cls.env.get(env, False)

    @staticmethod
    def match_or_eq(x, y):
        """ Matches two things if they can be matched. If not, it checks if they are equal. """
        if isinstance(x, Matchable):
            return x @ y
        if isinstance(y, Matchable):
            return y @ x
        if x == y:
            return x
        raise Mismatch

    @property
    def r(self):
        return self  # why??

    @classmethod
    def check_and_convert(cls, args, uuid, **kwargs):
        """Check and convert"""
        if not cls._can_be_converted(args, **kwargs):
            raise ValueError('Cannot convert.')
        return cls._convert(args, uuid, **kwargs)

    @staticmethod
    def _can_be_converted(args, **kwargs):
        """Check if can be converted"""
        return True

    @staticmethod
    def _convert(args, uuid, **kwargs):
        """Convert"""
        raise NotImplementedError

    @classmethod
    def all_subclasses(cls):
        """Return all sub classes as a set"""
        return set(cls.__subclasses__()).union([s
                                                for c in cls.__subclasses__()
                                                for s in c.all_subclasses()])

    @classmethod
    def convert(cls, args, uuid=None, **kwargs):
        """Convert args"""
        if uuid is None:
            if cls.context is None:
                uuid = str(uuid4())
            else:
                uuid = cls.context

        if isinstance(args, Builder):
            return args(uuid)

        for a_subclass in sorted(cls.all_subclasses(),
                                 key=lambda x: x.convert_precedence,
                                 reverse=True):
            if isinstance(args, a_subclass):
                return args
            try:
                return a_subclass.check_and_convert(args, uuid, **kwargs)
            except (ValueError, NotImplementedError):
                continue

        if not isinstance(args, str):
            return Mvar(args)
        elif args == '|':
            return '|'
        elif '.' in args:
            return float(args)
        elif args.startswith(':'):
            return args
        try:
            return int(args)
        except ValueError:
            raise ValueError('Cannot convert. Invalid value.')



convert = Matchable.convert


@contextmanager
def matchcontext():
    """Match context"""
    previous_context = Matchable.context
    try:
        Matchable.context = str(uuid4())
        yield
    finally:
        Matchable.context = previous_context


class _Nothing(Matchable):
    """ A *Matchable* that only matches with itself."""
    precedence = 400

    def match(self, other):
        """Match against other"""
        if other is not Nothing:
            raise Mismatch
        return self

    def __copy__(self):
        return self

    def copy(self, mappings, uuid=None):
        """Copy"""
        return self

    def __repr__(self):
        return 'Nothing'

    def __str__(self):
        return '-'


Nothing = _Nothing()


class Builder:
    """Builder"""
    def __init__(self, f):
        self.f = f

    def __call__(self, uuid):
        return self.f(uuid)


class Mvar(Modifiable, Matchable):
    """ A variable.

    *Mvar*s can have a value or be free. If they are free, they will
    take a value when matched with anything that's not a free *Mvar*.
    If matched with a free *Mvar*, both matches will behave
    as one from then on.

    >>> a = Mvar()
    >>> a  # free
    ?a
    >>> a @ 1  # the variable is being matched with 1
    1
    >>> a  # has taken the value 1 after matching
    1

    >>> a = Mvar()
    >>> b = Mvar()
    >>> a @ b  # a and b are now tied to each other
    ?a
    >>> b @ 1
    1
    >>> a
    1
    """
    d = {}
    v = {}
    name_mapping = {}
    i = 0

    def __new__(cls, val=None, id_=None, **kwargs):
        if id_ in Mvar.v:
            # get object from weakref
            x = Mvar.v[id_]()
            if x is not None:
                return x
        return super().__new__(cls)

    def __init__(self, val=None, id_=None, instantiation_callbacks=None):
        if id_ in Mvar.v:
            # assert(val is None or val == self.val)
            # get object from weakref
            x = Mvar.v[id_]()
            if x is not None:
                return
        if id_ is None:
            self.id_ = str(uuid4())
        else:
            self.id_ = id_

        Mvar.i += 1
        if self.id_ not in Mvar.d or val is not None:
            if self.id_ in Mvar.d and val is not None:
                ...
            # assert(val == Mvar.d[self.id_])
            Mvar.d[self.id_] = val
        # weakref is needed to make garbage collection work
        Mvar.v[self.id_] = weakref.ref(self)
        self.parent = None
        self.instantiation_callbacks = instantiation_callbacks or []
        self._instantiation_values = {}

    @property
    def precedence(self):
        """Return precedence"""
        if self.free:
            return 500
        return precedence(self.rval)

    @property
    def id(self):
        """Return id"""
        if self.parent is None:
            return self.id_
        return self.parent.id

    @property
    def val(self):
        """Return the value"""
        return Mvar.d[self.id]

    @val.setter
    def val(self, value):
        """Set the value"""
        Mvar.d[self.id] = value

    @property
    def rval(self):
        """Return the r value"""
        try:
            return self.val.rval
        except AttributeError:
            return self.val

    @property
    def r(self):
        """Return the r member variable"""
        if self.parent is not None:
            return self.parent.r
        if self.free:
            return self
        return self.val.r

    def redirect(self, other):
        """Redirect, used for debugging"""
        if DEBUG[0]:
            from IPython.core.debugger import set_trace
            set_trace()
        if self.id == other.id:
            return
        ics = self.instantiation_callbacks + other.instantiation_callbacks
        self.parent = other
        self.instantiation_callbacks = ics
        other.instantiation_callbacks = ics
        Mvar.v[self.id_] = weakref.ref(other)
        try:
            del Mvar.d[self.id_]
        except KeyError:
            pass

    @property
    def free(self):
        """Free the value"""
        return self.val is None or free(self.val)

    @property
    def partly_free(self):
        """Partly free the value"""
        return self.free or (hasattr(self.val, 'partly_free') and self.val.partly_free)

    @classmethod
    def by_id(cls, id_):
        """Return value by id_"""
        try:
            return cls.d[id_]
        except KeyError:
            return None

    def _redirect_to_smaller_id(self, other):
        hid = min(hash(self.id), hash(other.id))
        if hash(self.id) == hid:
            other.redirect(self)
        else:
            self.redirect(other)
        return

    def match(self, other):
        """Match against other"""
        try:
            if self == other:
                return self
        except ValueError:
            pass
        if not isinstance(other, Mvar):
            for a_ic in self.instantiation_callbacks:
                output = a_ic(other)
                if isinstance(output, dict):
                    self._instantiation_values.update(output)
            self.instantiation_callbacks = []
            self.val = other
            return self
        if self.free and other.free:
            self._redirect_to_smaller_id(other)
            return self
        elif self.free and not other.free:
            for a_ic in self.instantiation_callbacks:
                a_ic(other.rval)
            self.instantiation_callbacks = []
            self.redirect(other)
            return self
        elif self.free:
            self.redirect(other)
            return self
        raise Mismatch

    def repr_(self):
        """Return string representation of Mvar"""
        return f'Mvar({repr(self.val)}, {repr(self.id_)})'

    def __repr__(self):
        if self.val is None:
            id_ = self.name_mapping.get(self.id, self.id)
            try:
                return f'...{id_.split("...")[1]}'
            except (AttributeError, IndexError):
                pass
            try:
                id_ = id_.split('?')[1]
            except (AttributeError, IndexError):
                pass
            return f'?{id_}'
        return f"{self.val}"

    def repr(self):
        """Return string representation of variable"""
        if self.val is None:
            id_ = self.id
            try:
                return f'...{id_.split("...")[1]}'
            except (AttributeError, IndexError):
                pass
            try:
                id_ = id_.split('?')[1]
            except AttributeError:
                pass
            return f"'?{id_}'"
        return repr(self.val)

    def __eq__(self, other):
        if not self.free and self.val == val(other):
            return True
        if isinstance(other, Mvar):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    def copy(self, mappings, uuid=None):
        """Copy variable"""
        try:
            val = self.val.copy(mappings, uuid)
        except AttributeError:
            val = self.val
        if self.id not in mappings:
            ics = [i.copy(mappings, uuid) for i in self.instantiation_callbacks]
            mappings[self.id] = Mvar(val, instantiation_callbacks=ics)
        return mappings[self.id]

    def __del__(self):
        try:
            del Mvar.d[self.id_]
            del Mvar.v[self.id_]
        except KeyError:
            pass

    @staticmethod
    def _convert(args, uuid, **kwargs):
        """Convert"""
        if isinstance(args, str) and ' ' in args.strip():
            raise ValueError
        if isinstance(args, str) and args.strip() == '?':
            return Mvar(**kwargs)
        if isinstance(args, str) and args.startswith('?'):
            return Mvar(id_=uuid + args, **kwargs)
        elif isinstance(args, str) and args.startswith('...'):
            if len(args) == 1:
                return Mvar(*kwargs)
            return Mvar(id_=uuid + '?' + args, *kwargs)
        raise ValueError

    class simple_names:
        def __init__(self, vars_, name_all=False):
            vars_ = [v for v in vars_ if isinstance(v, Mvar) and v.free]
            self.mapping = self.map(vars_, name_all)

        def __enter__(self):
            Mvar.name_mapping = self.mapping

        def __exit__(self, type, value, traceback):
            Mvar.name_mapping = {}

        @staticmethod
        def map(vars_, name_all=False):
            """Return mapping"""
            mapping = {}
            letters = (str(x) for x in itertools.chain('abcdefghijklmnopqrstuvwxyz',
                                                       itertools.count()))
            for a_value in set([v.id for v in vars_]):
                if not name_all and vars_.count(a_value) == 1:
                    mapping[a_value] = ''
                else:
                    mapping[a_value] = next(letters)

            return mapping


class Matchall(Matchable):
    """ A *Matchable* that simple matches everything."""
    convert_precedence = 2000
    precedence = 2000

    @staticmethod
    def _can_be_converted(args, **kwargs):
        return isinstance(args, str) and args.strip() == '??'

    @classmethod
    def _convert(cls, args, uuid, **kwargs):
        return cls()

    def match(self, other):
        """Match against other"""
        return other

    def repr_(self):
        """String representation"""
        return 'Matchall()'

    def __repr__(self):
        return '??'

    def copy(self, mappings, uuid=None):
        """Copy"""
        return Matchall()


class Msequence(Matchable):
    """A sequence"""
    precedence = 100

    @staticmethod
    def _match(self, other):
        if not isinstance(other, Msequence):
            raise Mismatch
        if not len(self) == len(other):
            raise Mismatch
        for x, y in zip(self, other):
            x @ y
        return self

    def match(self, other):
        """Match against other"""
        mappings = {}
        self_ = self.copy(mappings)
        other_ = other.copy(mappings)
        self._match(self_, other_)
        self._match(self, other)

    def __setitem__(self, key, value):
        if key in self:
            self[key] @ value
        else:
            super().__setitem__(key, value)

    def empty(self):
        if self:
            raise Mismatch

    def __copy__(self):
        my_dict = {}
        for a_value in set(self):
            my_dict[a_value] = self.copy(a_value)
        return self.__class__([my_dict[x] for x in self])

    def copy(self, mappings, uuid=None):
        """Copy"""
        return self.__class__([copy(x, mappings, uuid) for x in self])

    def __hash__(self):
        return sum([hash(x) * 10 ** i for i, x in enumerate(self)])

    def __eq__(self, other):
        try:
            return all([a == b for a, b in zip(self, other)])
        except TypeError:
            return False

    def repr(self):
        """String representation"""
        return f'{type(self).__name__}([{",".join([repr_(x) for x in self])}])'


class Mlist(Msequence, list):
    """A list"""
    def head(self):
        """Return first item"""
        return self[0]

    def rest(self):
        """Return remaining items"""
        return Mlist(self[1:])


class Mdict(Msequence, dict):
    """A dictionary"""
    def __init__(self, val_, strict=False):
        self.strict = strict
        super().__init__(val(val_))

    @staticmethod
    def _can_be_converted(x, **kwargs):
        return isinstance(x, dict)

    @staticmethod
    def _convert(args, uuid, **kwargs):
        return Mdict({k: Matchable.convert(v, uuid) for k, v in args.items()})

    @staticmethod
    def _match(self, other):
        if not isinstance(other, Mdict):
            raise Mismatch
        for k in list(self.keys()) + list(other.keys()):
            if k not in self:
                if self.strict:
                    raise Mismatch
                self[k] = other[k]
            if k not in other:
                if self.strict:
                    raise Mismatch
                other[k] = self[k]
        if not len(self) == len(other):
            raise Mismatch
        for k in self:
            Matchable.match_or_eq(self[k], other[k])
        return self

    def __copy__(self):
        return self.__class__({k: self.copy(v) for k, v in self.items()})

    def copy(self, mappings, uuid=None):
        return self.__class__({k: copy(v, mappings) for k, v in self.items()})

    def __hash__(self):
        return sum([hash(x) * 10 ** i for i, x in enumerate(self.values())])

    def __eq__(self, other):
        if not isinstance(other, Mdict):
            return False
        if not len(self) == len(other):
            return False
        return all([self[k] == other[k] for k in self])

    def repr(self):
        """String representation"""
        return f'{type(self).__name__}' \
               f'({{{",".join([k + ":" + repr_(v) for k, v in self.items()])}}})'


class Mtuple(Msequence, tuple):
    """A tuple"""
    @staticmethod
    def _can_be_converted(x, **kwargs):
        return False

    @staticmethod
    def _convert(args, uuid, **kwargs):
        return Mtuple([Matchable.convert(x, uuid) for x in args])


class Mfun(Modifiable, Matchable):
    """A function"""
    precedence = 100
    convert_precedence = 100

    def __init__(self, var, fun, inv, **kwargs):
        self.fun = fun
        self.inv = inv
        self.var = var
        self.kwargs = kwargs

    @property
    def val(self):
        """Return value of the function"""
        if self.var.rval is not None:
            return self.fun(self.var.rval)
        return None

    @property
    def vars(self):
        """Return var as a list"""
        return [self.var]

    @property
    def rval(self):
        """Return val member variable"""
        return self.val

    @property
    def partly_free(self):
        """Return partly_free member variabel"""
        return self.var.partly_free

    @property
    def r(self):
        if self.partly_free:
            return self
        return self.rval

    def match(self, other):
        """Match against other"""
        if isinstance(other, Mfun):
            if self.identical(other):
                return self

            if self.partly_free and other.partly_free and self.var.id == other.var.id:
                raise Mismatch

            self.var @ Mfun(
                other.var,
                lambda x: self.inv(other.fun(x)),
                lambda x: other.inv(self.fun(x))
            )

            if self.var.rval is None:
                return self
            return self.fun(self.var.rval)

        if free(other):
            raise Mismatch

        return self.fun(val(self.var @ self.inv(val(other))))

    def __repr__(self):
        if self.val is not None:
            return repr(self.val)
        return f'{self.fun} {self.val}'

    def identical(self, other):
        """Check if identical compared to other"""
        return (isinstance(other, Mfun)
                and self.var.id == other.var.id
                and self.fun.__code__ == other.fun.__code__
                and self.kwargs == other.kwargs)

    def repr(self):
        """String representation"""
        return f'"{repr(self)}"'

    def __eq__(self, other):
        sval = val(self)
        oval = val(other)

        if sval is None or oval is None:
            return False

        return sval == oval

    def __hash__(self):
        return hash(self.var) + hash(self.fun) + hash(self.inv) + hash(str(self.kwargs))


class Madd(Mfun):
    """Addition"""
    def __init__(self, var, a):
        if not isinstance(val(a), (int, float)):
            raise TypeError('Arg must be a number.')
        self.a = a
        super().__init__(var, lambda x: x + a, lambda x: x - a, a=a)

    @staticmethod
    def _can_be_converted(x, **kwargs):
        return isinstance(x, str) and '+' in x

    @staticmethod
    def _convert(args, uuid, **kwargs):
        a_arg, b_arg = args.split('+')
        return Matchable.convert(a_arg.strip(), uuid) + Matchable.convert(b_arg.strip(), uuid)

    def __repr__(self):
        if self.val is not None:
            return repr(self.val)
        return f'{self.var} + {self.a}'

    def copy(self, mappings, uuid=None):
        """Copy"""
        return Madd(self.var.copy(mappings), copy(self.a, mappings))


class Msub(Mfun):
    """Subtraction"""
    def __init__(self, var, a):
        if not isinstance(val(a), (int, float)):
            raise TypeError('Arg must be a number.')
        self.a = a
        super().__init__(var, lambda x: x - a, lambda x: x + a)

    @staticmethod
    def _can_be_converted(x, **kwargs):
        return isinstance(x, str) and '-' in x

    @staticmethod
    def _convert(args, uuid, **kwargs):
        a_arg, b_arg = args.split('-')
        return Matchable.convert(a_arg.strip(), uuid) - Matchable.convert(b_arg.strip(), uuid)

    def __repr__(self):
        if self.val is not None:
            return repr(self.val)
        return f'{self.var} - {self.a}'

    def copy(self, mappings, uuid=None):
        """Copy"""
        return Msub(self.var.copy(mappings), copy(self.a, mappings))


class Mmul(Mfun):
    """Multiplication"""
    def __init__(self, var, a):
        if not isinstance(val(a), (int, float)):
            raise TypeError('Arg must be a number.')
        self.a = a
        super().__init__(var, lambda x: x * a, lambda x: x / a)

    @staticmethod
    def _can_be_converted(x, **kwargs):
        return isinstance(x, str) and '*' in x

    @staticmethod
    def _convert(args, uuid, **kwargs):
        a, b = args.split('*')
        return Matchable.convert(a.strip(), uuid) * Matchable.convert(b.strip(), uuid)

    def __repr__(self):
        if self.val is not None:
            return repr(self.val)
        return f'{self.var} * {self.a}'

    def copy(self, mappings, uuid=None):
        """Copy"""
        return Mmul(self.var.copy(mappings), copy(self.a, mappings))


class Mdiv(Mfun):
    """Division"""
    def __init__(self, var, a):
        if not isinstance(val(a), (int, float)):
            raise TypeError('Arg must be a number.')
        self.a = a
        super().__init__(var, lambda x: x / a, lambda x: x * a)

    def __repr__(self):
        if self.val is not None:
            return repr(self.val)
        return f'{self.var} / {self.a}'

    def copy(self, mappings, uuid=None):
        """Copy"""
        return Mdiv(self.var.copy(mappings), copy(self.a, mappings))


class InstantiationCallback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def copy(self, mappings, uuid):
        return self


class ListTemplateCallback:
    def __init__(self, template):
        self.template = template

    def __call__(self, other):
        mappings = {}
        temp_copy = self.template.copy(mappings)
        for i, v in enumerate(other.copy(mappings)):
            Matchable.match_or_eq(v, temp_copy[i])
        other.template = self.template

    def copy(self, mappings, uuid):
        return ListTemplateCallback(self.template.copy(mappings, uuid))


class ListLenCallback:
    def __init__(self, _len):
        self._len = _len

    def __call__(self, other):
        other._len @ self._len

    def copy(self, mappings, uuid):
        return ListLenCallback(copy(self._len, mappings, uuid))


class Template:
    def __init__(self, *args, start=0, uuid=None, _cache=None):
        self.start = start
        if uuid is None:
            self.uuid = str(uuid4())
        else:
            self.uuid = uuid

        self.args = args
        if _cache is None:
            self._cache = {}
        else:
            self._cache = _cache

    def fromstart(self, start):
        return type(self)(
            *self.args,
            start=self.start + start,
            uuid=self.uuid,
            _cache=self._cache
        )

    def copy(self, mappings, uuid=None):
        if uuid is None:
            uuid = str(uuid4())
        ul = len(self.uuid)
        suuid = self.uuid[:ul//2] + uuid[:ul//2]
        args = [copy(arg, mappings, uuid) for arg in self.args]
        return type(self)(*args, start=self.start, uuid=suuid,
                          _cache={k: copy(v, mappings, uuid) for k, v in self._cache.items()})

    def __getitem__(self, item):
        if item + self.start not in self._cache:
            self._cache[item + self.start] = self._getitem(item)
        return self._cache[item + self.start]

    def _getitem(self, item):
        raise NotImplementedError

    def __repr__(self):
        return type(self).__name__


class Fixed(Template):
    def __init__(self, template, start=0, uuid=None, _cache=None):
        self.template = template
        super().__init__(template, start=start, uuid=uuid, _cache=None)

    def _getitem(self, item):
        return self.template


class Gen(Template):
    def __init__(self, gen_function, start=0, uuid=None, _cache=None):
        self.gen_function = gen_function
        super().__init__(gen_function, start=start, uuid=uuid, _cache=None)

    def _getitem(self, item):
        return self.gen_function(item)


class Counter(Template):
    """Counter"""
    def _getitem(self, item):
        return item + self.start


class NamedVars(Template):
    """Named variables"""
    def __init__(self, modifier=None, start=0, uuid=None, _cache=None):
        self.modifier = modifier
        super().__init__(modifier, start=start, uuid=uuid, _cache=_cache)

    def _getitem(self, item):
        var = Matchable.convert(f'?{item + self.start}', uuid=self.uuid)
        if self.modifier is None:
            return var
        return self.modifier(var)


class Lsequence(Matchable):
    """ A variable length sequence.

    A Lsequence has a *head* and a *tail*. The *head* is the sequences element at position 0,
    the *tail* is another Lsequence that represents the rest of the sequence.

    >>> x = Llist.fromlist([1,2,3,4,5])
    >>> x.head
    1
    >>> x.tail
    [2,3,4,5]

    A Llist can be created by sending a list to Matchable.convert.
    Entries are converted recursively.

    >>> Matchable.convert([1, 2, 3])
    [1, 2, 3]
    >>> Matchable.convert(['?a', '?b', '?c'])
    [?a, ?b, ?c]

    A free tail is represented by '...'
    >>> Matchable.convert(['?a', '?b', '...'])
    [?a, ?b, ...]

    It can also be named (for matching with another Llist.
    >>> Matchable.convert(['?a', '?b', '...a'])
    [?a, ?b, ...a]

    The *tail* can be free (-> FreeLsequence), which means that the length of the sequence is
    not yet determined (it may be determined when matching the sequence with other sequences).

    The empty Lsequence has `head == Nothing` and `tail == Nothing`.

    A Lsequence can have a *template*. The *template* is a *Template*-object and it is matched
    with each entry of the sequence.

    >>> Llist(Mvar(), Llist(Mvar(), Llist(Nothing(), Nothing())))
    ['?a', '?b']
    >>> Llist(Mvar(), Llist(Mvar(), Llist(Nothing(), Nothing())), template=Fixed(1))
    [1, 1]
    >>> Llist(Mvar(), Llist(Mvar(), Llist(Nothing(), Nothing())), template=Counter())
    [1, 2]

    A Lsequence has a length - this will be a free Mvar in the case of a list with free tail.

    >>> a = Mvar()
    >>> a
    ?
    >>> b = Matchable.convert([1, 2, 3], len_=a)
    >>> a
    3
    """
    @property
    def precedence(self):
        """Return the precedence"""
        if self.head is Nothing:
            return 200
        return 100

    def __init__(self, head, tail=None, template=None, len_=None):
        self._len = Mvar(len_)
        self.x = Mtuple((head, tail))
        self.uuid = str(uuid4())
        self.template = template

        # what is `template` for?

    @property
    def template(self):
        """Return the template"""
        return self._template

    @template.setter
    def template(self, val_):
        # template matches all (-> basically no template)
        if val_ is None:
            self._template = Fixed(Matchall())
            return
        # "fixed" template (all values must match `val_`)
        if not isinstance(val_, Template):
            val_ = Fixed(val_)

        # at this stage, val_ is a Template
        self._template = val_

        if self.empty:
            return

        # match the template with existing values
        Matchable.match_or_eq(self.head, val_[0])
        if val(self.tail) is not None:
            val(self.tail).template = val_.fromstart(1)
        elif free(self.tail):
            r(self.tail).instantiation_callbacks += [ListTemplateCallback(val_.fromstart(1))]

    def check_len(self):
        """Check the length"""
        a_len = val(self.len)
        if not a_len:
            return
        self @ type(self)._convert([f'...{a_len}x'], self.uuid, self.template)

    @property
    def len(self):
        if free(self.tail):
            return self._len
        if self.empty:
            return self._len @ 0
        return self._len @ (val(self.tail).len + 1)

    @classmethod
    def _can_be_converted(cls, args, **kwargs):
        return False

    @classmethod
    def _convert(cls, args, uuid, template=None, len_=None, **kwargs):
        # args is either a list (Llist) or a tuple (Ltuple) because of their _can_be_converted
        args = list(args)

        my_list = []
        for x in args:
            # matches '...0a', '...1', '...10street'
            # '...<n>a' is a shortcut for adding n variable items '?a0', ..., '?an'
            a_match = isinstance(x, str) and re.match('\.\.\.([0-9]+)(.*)', x)
            # if there is a match ...
            if a_match:
                l_ = int(a_match.groups()[0])
                name = a_match.groups()[1]
                # ... add n variables '?<name><i>' instead of '...<name><n>'
                for i in range(l_):
                    my_list.append((f'?{name}{i}'))
            else:
                my_list.append(x)

        # create a list with free tail if last item of l starts with '...'
        try:
            if len(my_list) and my_list[-1].startswith('...'):
                return cls.fromlist(
                    [Matchable.convert(x, uuid) for x in my_list[:-1]],
                    tail=cls.free_version(template=template, len_=len_,
                                          id_=uuid + '?' + my_list[-1])
                )
        except (TypeError, AttributeError):
            pass

        return cls.fromlist([Matchable.convert(x, uuid) for x in my_list], template=template,
                            len_=len_)

    @property
    def vars(self):
        v = []
        try:
            v += self.head.vars
        except AttributeError:
            v.append(self.head)
        try:
            v += self.tail.vars
        except AttributeError:
            v.append(self.tail)
        return v

    @classmethod
    def fromlist(cls, l, tail=None, template=None, len_=None):
        if len(l) == 0:
            if tail is None:
                return cls(Nothing, Nothing)
            return tail
        if l[0] == '|':
            return l[1]
        res = cls(l[0], cls.fromlist(l[1:], tail=tail), template=template, len_=len_)
        return res

    @property
    def head(self):
        """Return the head"""
        return self.x[0]

    @property
    def tail(self):
        """Return the tail"""
        return self.x[1]

    @property
    def empty(self):
        """Check if empty"""
        return val(self.head) is Nothing and val(self.tail) is Nothing

    def copy(self, mappings, uuid=None):
        """Copy"""
        template = self.template.copy(mappings, uuid)
        len_ = self._len.copy(mappings, uuid)
        return type(self)(*self.x.copy(mappings, uuid), len_=len_, template=template)

    def infinite_iter(self):
        """ Iterate over the sequence (potentially infinitely if the tail is free). """
        if self.empty:
            return
        yield self.head
        if isinstance(self.tail, Mvar):
            r = type(self)(Mvar(), Mvar())
            self.tail @ r
            yield from r
        else:
            yield from self.tail

    def __add__(self, other):
        """ Concatenate two sequences. """
        if self.empty:
            return other
        return type(self)(self.head, self.tail + other)

    def iter_no_tail(self):
        for x in self:
            if isinstance(x, tuple) and len(x) and x[0] == 'tail':
                return
            yield x

    def __iter__(self):
        if self.empty:
            return
        if self.head is not Nothing:
            yield self.head
        if isinstance(self.tail, Mvar):
            tail = self.tail.rval
            if tail is None:
                tail = self.tail
        else:
            tail = self.tail
        if tail is self:
            raise ValueError('List contains a cycle.')
        try:
            yield from tail
        except TypeError:
            yield 'tail', tail
            return

    def _match(self, other):
        # other must be of same type
        if type(other) != type(self):
            raise Mismatch
        # both empty -> match
        if self.empty and other.empty:
            return self
        # no head -> tail must match other
        if self.head is Nothing:
            return self.tail @ other
        if other.head is Nothing:
            return self @ other.tail

        # mainly:
        # head must match
        Matchable.match_or_eq(self.head, other.head)
        with Matchable.set_env('nested'):
            # and tail must match
            self.tail @ other.tail
            Matchable.match_or_eq(self.len, other.len)
        return self

    def match(self, other):
        """Pattern match. """
        mappings = {}
        # before really matching variables, check if they can match by matching copies of
        # self and other
        if not Matchable.in_env('testing') and not Matchable.in_env('nested'):
            with Matchable.set_env('testing'):
                scopy = self.copy(mappings)
                ocopy = copy(other, mappings)
                scopy._match(ocopy)
        self._match(other)
        return self

    def _get_single_item(self, item: int):
        """Get a single item (i.e. no slice)."""
        assert item >= 0
        if self.empty:
            raise IndexError
        if item == 0:
            return self.head
        return val(self.tail)[item - 1]

    def _get_slice(self, item):
        """Get a slice of the list. """
        if item.step is not None and item.step != 1:
            raise ValueError('Lookup with slices only supported with step=1.')
        if self.empty:
            return type(self).fromlist([])
        start = item.start or 0
        if start == 0 and item.stop == 1:
            return type(self)(self.head, type(self)(Nothing, Nothing))
        if item.stop is not None and start >= item.stop:
            return type(self)(Nothing, Nothing)
        if item.stop is not None and start == 0:
            return type(self)(self.head, self.tail[:item.stop - 1])
        if start == 0:
            return self
        if val(self.tail) is not None:
            stop = item.stop
            if stop is not None:
                stop = stop - 1
            return val(self.tail)[item.start - 1:stop]
        raise ValueError('something wrong')

    def __getitem__(self, item):
        if not isinstance(item, slice):
            return self._get_single_item(item)

        return self._get_slice(item)

    def __str__(self):
        a_list = list(self)
        res = self.ldelimiter + ', '.join([str(x) for x in a_list[:-1]])
        if a_list and isinstance(a_list[-1], tuple) and a_list[-1][0] == 'tail':
            res += f' | {a_list[-1][1]}' + self.rdelimiter
        else:
            if len(a_list) > 1:
                res += ', '
            if a_list:
                res += f'{a_list[-1]}'
            res += self.rdelimiter
        return res

    def __repr__(self):
        return str(self)

    def repr(self):
        """String representation"""
        return self.__class__.__name__ + f'({repr_(self.head)}, {repr_(self.tail)})'

    def __eq__(self, other):
        if not isinstance(r(other), type(self)):
            return False
        return self.head == r(other).head and self.tail == r(other).tail

    def __hash__(self):
        return hash(self.repr())


class FreeLsequence(Mvar):
    def __init__(self, val=None, id_=None, template=None, len_=None):
        callbacks = []
        if len_ is not None:
            callbacks += [ListLenCallback(len_)]
        if template is not None:
            callbacks += [ListTemplateCallback(template)]
        self._template = template
        self.len_ = Mvar(len_)
        Mvar.__init__(self, id_=id_, instantiation_callbacks=callbacks)

    @property
    def precedence(self):
        if self.free:
            return 499
        return super().precedence

    def match(self, other):
        if not isinstance(other, self.base):
            raise Mismatch
        Mvar.match(self, other)
        Matchable.match_or_eq(self.len, other.len)
        return self

    @property
    def len(self):
        return self.len_

    def __hash__(self):
        return hash(self.id)

    @staticmethod
    def _can_be_converted(x, **kwargs):
        return False

    @property
    def head(self):
        raise TypeError('Free list has no head.')

    @property
    def tail(self):
        raise TypeError('Free list has no tail.')

    @property
    def empty(self):
        raise TypeError('List is free - cannot determine emptyness.')

    def infinite_iter(self):
        raise TypeError('Free list does not support iteration.')

    def __iter__(self):
        raise TypeError('Free list does not support iteration.')

    def iter_no_tail(self):
        raise TypeError('Free list does not support iteration.')

    def __getitem__(self, item):
        raise TypeError('Free list does not support getitem.')

    def __add__(self, other):
        raise TypeError('Cannot append to free list.')

    def __eq__(self, other):
        if self.free:
            return self is other
        return super().__eq__(other)

    def copy(self, mappings, uuid=None):
        if self.id not in mappings:
            if self.free:
                a_copy = type(self)(
                    template=copy(self._template, mappings, uuid),
                    len_=copy(self.len_, mappings, uuid)
                )
            else:
                a_copy = val(self).copy(mappings, uuid)
            mappings[self.id] = a_copy
        return mappings[self.id]

    def __repr__(self):
        if self.free:
            res = f'{self.ldelimiter}{Mvar.__repr__(self)}{self.rdelimiter}'
            if self._template is not None or self.len is not None:
                res += ' with '
            if self._template is not None:
                res += f'template={self._template} and '
            if self.len is not None:
                res += f'len={self.len}'
            return res
        else:
            return super().__repr__()

    def __str__(self):
        return self.__repr__()

    def repr(self):
        return self.__class__.__name__ + f'({repr_(self._template)}, {repr_(self.len_)})'

    @property
    def vars(self):
        if self.free:
            return [self]
        return val(self).vars


class Llist(Lsequence):
    """L-list"""
    ldelimiter = '['
    rdelimiter = ']'

    @staticmethod
    def _can_be_converted(x, **kwargs):
        return isinstance(x, list)


class Ltuple(Lsequence):
    """L-tuple"""
    ldelimiter = '('
    rdelimiter = ')'

    @staticmethod
    def _can_be_converted(x, **kwargs):
        return isinstance(x, tuple)


class FreeLlist(FreeLsequence, Llist):
    """Free L-list"""
    base = Llist


Llist.free_version = FreeLlist


class FreeLtuple(FreeLsequence, Ltuple):
    """Free L-tuple"""
    base = Ltuple


Ltuple.free_version = FreeLtuple
