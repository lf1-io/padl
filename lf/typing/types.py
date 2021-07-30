# pylint: disable=no-member,not-callable
"""Types"""
import json
import random
from string import printable
import numpy
import torch

import PIL
from .patterns import (
    Mvar, Mdict, Mismatch, uuid4, Mtuple, val, Ltuple, repr_, Matchable,
    Builder, Matchall, Llist, Fixed, Nothing, convert, free, Lsequence
)
from . import samplers, checkers


settings = {
    'debug': False
}


class TypeMismatch(Mismatch):
    """Type mismatch"""
    ...


class TypeBuilder(Builder):
    """Type builder"""
    ...


class Type:
    """A type.

    :param name: Name of the type.
    :param parent: Parent type.
    :param argnames: Argument names that can be used with the type.
    :param jit_checker: Function ? -> bool for checking if some input matches the type.
    :param sampler: Function taking the typevar as argument and generates a sample.
    """
    def __init__(self, name, parent=None, argnames=(),
                 jit_checker=lambda x, kwargs_: True, sampler=None,
                 serve_convert=lambda x: x):
        assert isinstance(name, str)
        self.name = name
        self.parent = parent
        self.argnames = argnames
        self.jit_checker = jit_checker
        self.sampler = sampler
        self.serve_convert = serve_convert

    def __le__(self, other):
        if self.name == other.name:
            return True
        if self.parent is not None:
            return self.parent <= other
        return False

    def __repr__(self):
        return f'Type({repr(self.name)}, {repr(self.parent)}, {repr(self.argnames)})'

    def __str__(self):
        a_name = self.name
        return a_name

    def var(self, name=None, uuid=None, **kwargs):
        """Create a typvar from the type. """
        return Typevar(self, name=name, uuid=uuid, **kwargs)

    def __call__(self, name=None, **kwargs):
        def type_builder(uuid):
            """Type builder"""
            kwargs_ = {}
            for k, values in kwargs.items():
                if callable(values):
                    kwargs_[k] = values(uuid)
                else:
                    kwargs_[k] = values
            return self.var(name=name, uuid=uuid, **kwargs_)
        return TypeBuilder(type_builder)

    def issubtype(self, other):
        """Check if is sub-type"""
        if other == self:
            return True
        if other.name == 'Any':
            return True
        if self.name == 'Any' and other.name != 'Any':
            return False
        return self.parent.issubtype(other)


class AnyType(Type):
    """Any type"""
    ...


class FunctionalType(Type):
    """
    Functional type
    """
    def __init__(self, f):
        self.f = f

    def __matmul__(self, other):
        return self.f(other)


class Typevar(Matchable):
    """ Type variable.

    :param type: type
    :param name: name
    :pram uuid:
    :param kwargs_:
    :param match_callback:
    :param kwargs: additional key word arguments
    """
    def __init__(self, type, name=None, uuid=None, kwargs_=None,
                 match_callback=None, **kwargs):
        self.uuid = uuid
        self.name = name

        if isinstance(type, Mvar):
            self.type = type
        else:
            type_ = Mvar.by_id(self.id_)
            if type_ is not None and type_ <= type:
                type = type_
            self.type = Mvar(type, id_=self.id_)

        if self.id_ is None:
            kid = None
        else:
            kid = self.id_ + '_k'

        if isinstance(self.type.rval, Ltuple):
            self.kwargs_ = Mvar(Mdict({}, strict=True))
        elif isinstance(type, Mvar) and val(type) is None:
            if kwargs_ is None:
                kwargs_ = {}
            self.kwargs_ = Mvar(Mdict(Matchable.convert(kwargs_)))
        elif kwargs_ is None:
            self.kwargs_ = Mvar(Mdict({
                k: Matchable.convert(kwargs.get(k, '?'), uuid=uuid)
                for k in self.type.rval.argnames
            }), id_=kid)
        else:
            self.kwargs_ = kwargs_

        for k in kwargs:
            assert(k in self.type.rval.argnames)
        self.match_callback = match_callback

    @property
    def id_(self):
        if self.uuid is not None and self.name is not None:
            id_ = self.name + '::' + self.uuid
        elif self.name is not None:
            id_ = self.name
        else:
            id_ = None
        return id_

    def sample(self):
        """Sample a value using the type's sampler. """
        if isinstance(val(self.type), Lsequence):
            return tuple(val(x).sample() for x in val(self.type))

        return val(self.type).sampler(self)

    def from_string(self, str_):
        """Evaluate string. """
        return eval(str_)

    @property
    def is_tuple_typevar(self):
        """Check if typevar is a tuple. """
        return isinstance(self.type.rval, Ltuple)

    @property
    def names(self):
        if self.is_tuple_typevar:
            res = []
            for t in self.type.rval:
                try:
                    res += val(t).names
                except AttributeError:
                    continue
            return res
        if self.name is None:
            return []
        return [self.name]

    def jit_check(self, x):
        """Jit checker. """
        typeval = val(self.type)
        if isinstance(typeval, Ltuple):
            for x_val, y in zip(x, typeval.iter_no_tail()):
                y.jit_check(x_val)
            return
        typeval.jit_checker(x, self.kwargs_)

    @property
    def ids(self):
        if self.is_tuple_typevar:
            res = []
            for t in self.type.rval:
                try:
                    res += val(t).ids
                except AttributeError:
                    continue
            return res
        if self.id_ is None:
            return []
        return [self.id_]

    def __getitem__(self, item):
        if self.is_tuple_typevar:
            return val(self.type)[item]
        return val(self.kwargs)[item]

    def __contains__(self, item):
        try:
            return item in val(self.kwargs)
        except AttributeError:
            return False

    @property
    def vars(self):
        vars_list = []
        for v in val(self.kwargs_).values():
            try:
                vars_list += val(v).vars
            except AttributeError:
                pass
            vars_list.append(v)
            try:
                vars_list += v.vars
            except AttributeError:
                pass
        if self.is_tuple_typevar:
            res = []
            for t in self.type.rval:
                try:
                    res += val(t).vars
                except AttributeError:
                    res.append(t)
            return res
        return vars_list

    @property
    def kwargs(self):
        """Return kwargs"""
        return self.kwargs_.rval

    def remove_name(self, name):
        """
        Remove name

        :param name: name
        """
        if self.is_tuple_typevar:
            for t in self.type.rval:
                try:
                    t.remove_name(name)
                except AttributeError:
                    continue
        if self.name == name:
            self.name = None

    def match(self, other):
        """
        Match against other
        """
        if isinstance(other, Eithervar):
            return other @ self
        # apply match callbacks
        if self.match_callback is not None:
            self.match_callback(other)
        if other.match_callback is not None:
            other.match_callback(self)
        # if own type is free, match with other type
        if self.type.val is None:
            self.type @ other.type
            self.kwargs_ @ other.kwargs_
        # if other type is free assign to own type
        if other.type.val is None:
            other.type.redirect(self.type)
            self.kwargs_ @ other.kwargs_
        elif isinstance(val(self.type), Type) and val(self.type).name == 'Any':
            self.type.redirect(other.type)
            self.kwargs_ @ other.kwargs_
        elif isinstance(val(other.type), Type) and val(other.type).name == 'Any':
            other.type.redirect(self.type)
            self.kwargs_ @ other.kwargs_
        elif self.is_tuple_typevar or other.is_tuple_typevar:
            return self.type @ other.type
        elif self.type.val >= other.type.val:
            self.type.redirect(other.type)
            self.kwargs_ @ other.kwargs_
        elif other.type.val >= self.type.val:
            other.type.redirect(self.type)
            self.kwargs_ @ other.kwargs_
        else:
            raise Mismatch
        return self

    @staticmethod
    def match_all(typevars: list):
        """Match all"""
        t = typevars[0]
        for next_typevar in typevars[1:]:
            t = t @ next_typevar
        return t

    def __repr__(self):
        return self.repr(self.vars)

    def repr(self, vars=None):
        """Return string representation"""
        if vars is None:
            return self.repr_()
        keep = [v for v in set(vars)
                if not isinstance(v, Mvar) or not v.free or vars.count(v) > 1]
        with Mvar.simple_names(keep, True):
            if self.is_tuple_typevar:
                return f'{self.type.val}'
            type_ = self.type.val
            if type_ is None:
                n = '*'
            else:
                n = type_.name.lower()
            if self.name is not None:
                n += f'({self.name})'
            key_vals = [(k, v) for k, v in self.kwargs.items() if v in keep]
            if self.kwargs and key_vals:
                n += '[' + ','.join([f'{k}:{v}' for k, v in key_vals]) + ']'
        return n

    def repr_(self):
        """String representation"""
        return (f'Typevar({repr_(self.type)}, {repr(self.name)}, {repr(self.uuid)}{", " if self.kwargs else ""}'
                f'{", ".join([f"{k}={repr_(v)}" for k, v in self.kwargs.items()])})')

    def copy(self, mappings=None, uuid=None):
        """Copy"""
        if uuid is None:
            uuid = str(uuid4())
        if mappings is None:
            mappings = {}

        return Typevar(self.type.copy(mappings, uuid), self.name, uuid,
                       kwargs_=self.kwargs_.copy(mappings, uuid))

    @staticmethod
    def remove_single_names(typevars):
        names = sum([t.names for t in typevars], [])
        ids = sum([t.ids for t in typevars], [])
        for n, id_ in zip(names, ids):
            if ids.count(id_) == 1:
                for t in typevars:
                    t.remove_name(n)

    @classmethod
    def build(cls, type_desc, uuid=None):
        if uuid is None:
            uuid = str(uuid4())
        if isinstance(type_desc, Typevar):
            return type_desc
        elif isinstance(type_desc, tuple):
            return Typevar(Matchable.convert(type_desc, uuid=uuid))
        elif isinstance(type_desc, dict):
            return Typevar(Matchable.convert(tuple(type_desc.values()), uuid=uuid))
        elif isinstance(type_desc, TypeBuilder):
            return type_desc(uuid)
        elif isinstance(type_desc, Mvar):
            return Typevar(type_desc)
        raise ValueError('cannot build typevar')


def either(*typegens):
    def gen(uuid):
        typevars = []
        for t in typegens:
            if isinstance(t, Typevar):
                typevars.append(t)
            else:
                typevars.append(t(uuid))
        return Eithervar(typevars)
    return TypeBuilder(gen)


class Eithervar(Matchable):
    def __init__(self, typevars):
        self.typevars = typevars
        self.decided = None
        self.match_callback = None

    @property
    def names(self):
        return sum([x.names for x in self.typevars], [])

    @property
    def ids(self):
        return sum([x.ids for x in self.typevars], [])

    @property
    def vars(self):
        return sum([x.vars for x in self.typevars], [])

    def remove_name(self, n):
        for t in self.typevars:
            t.remove_name(n)

    def match(self, other):
        if self.decided is not None:
            return self.decided @ other
        for tv in self.typevars:
            try:
                self.decided = tv @ other
                return self.decided
            except Mismatch:
                continue
        raise Mismatch

    def __repr__(self):
        if self.decided:
            return self.decided.__repr__()
        return f'Either({[t.__repr__() for t in self.typevars]})'

    def repr_(self):
        """String representation"""
        return f'either({",".join(x.repr_() for x in self.typevars)})'

    def repr(self, vars=None):
        """String representation"""
        if vars is None:
            return self.repr_()
        return f'either({", ".join(x.repr(vars) for x in self.typevars)})'

    def copy(self, mappings=None, uuid=None):
        """Copy"""
        if uuid is None:
            uuid = str(uuid4())
        if mappings is None:
            mappings = {}
        return Eithervar([t.copy(mappings, uuid) for t in self.typevars])


class Ftype(Matchable):
    """Function type.

    :param x: The input type.
    :param y: The output type.
    :param constructor: If not *None*, apply this to *x* and *y* upon init.
    """

    def __init__(self, x, y, constructor=None):
        if isinstance(x, Typevar) and isinstance(y, Typevar):
            if x.id_ is not None and y.id_ is not None and x.id_ == y.id_:
                x @ y
        self.xy = Mtuple((x, y))
        if constructor is not None:
            constructor(x, y)
        self.constructor = constructor

    def remove_single_names(self):
        Typevar.remove_single_names([self.x, self.y])

    @property
    def names(self):
        return self.x.names + self.y.names

    @property
    def ids(self):
        return self.x.ids + self.y.ids

    @property
    def vars(self):
        return self.x.vars + self.y.vars

    def repr(self, vars=None):
        return f'{self.x.repr(vars)} ⟶ {self.y.repr(vars)}'

    @property
    def x(self):
        return self.xy[0]

    @property
    def y(self):
        return self.xy[1]

    def __repr__(self):
        vars = self.x.vars + self.y.vars
        x = f'({self.x.repr(vars)})' if isinstance(self.x, Ftype) else self.x.repr(vars)
        y = f'({self.y.repr(vars)})' if isinstance(self.y, Ftype) else self.y.repr(vars)
        return f'{x} ⟶ {y}'

    def to_dict(self):
        return {
            'x': self.x.to_dict() if isinstance(self.x, Ftype) else self.x.repr_(),
            'y': self.y.to_dict() if isinstance(self.y, Ftype) else self.y.repr_()
        }

    @classmethod
    def from_dict(cls, d):
        """
        Build from dictionary

        :param d: dictionary
        """
        if 'x' not in d and 'y' not in d:
            return FtypeDict.from_dict(d)
        x, y = d['x'], d['y']
        if isinstance(x, dict):
            x = cls.from_dict(x)
        else:
            x = eval(x)
        if isinstance(y, dict):
            y = cls.from_dict(y)
        else:
            y = eval(y)
        return cls.build(x, y)

    @classmethod
    def in_out_from_dict(cls, d):
        x, y = d['x'], d['y']
        if isinstance(x, dict):
            x = cls.from_dict(x)
        else:
            x = eval(x)
        if isinstance(y, dict):
            y = cls.from_dict(y)
        else:
            y = eval(y)

        return x, y

    def __matmul__(self, other):
        self.x @ other.x
        self.y @ other.y
        return self

    def copy(self, mappings=None, uuid=None):
        """Copy"""
        if uuid is None:
            uuid = str(uuid4())
        if mappings is None:
            mappings = {}
        return Ftype(*self.xy.copy(mappings, uuid), self.constructor)

    @staticmethod
    def build(in_type, out_type, clean=True):
        """
        Build

        :param in_type: in type
        :param out_type: out type
        :param clean: If True remove single names
        """
        uuid = str(uuid4())
        f_type = Ftype(Typevar.build(in_type, uuid), Typevar.build(out_type, uuid))
        if clean:
            f_type.remove_single_names()
        return f_type

    def __eq__(self, other):
        return str(self) == str(other)


class FtypeDict(dict):
    def copy(self, mappings=None, uuid=None):
        return FtypeDict({k: v.copy(mappings, uuid)
                          for k, v in self.items()})

    def to_dict(self):
        """
        Convert to dict
        """
        return {
            k: v.to_dict()
            for k, v in self.items()
        }

    def from_dict(self, d):
        """
        Build from dict

        :param d: dictionary
        """
        return {
            k: Ftype.from_dict(v)
            for k, v in d.items()
        }


def Ttuple(type_desc, template=None, len_=None):
    def build(uuid):
        if template is not None:
            template_ = Fixed(Matchable.convert(template, uuid))
        else:
            template_ = template
        if len_ is not None:
            len__ = Matchable.convert(len_, uuid)
        else:
            len__ = len_
        return Typevar(Matchable.convert(type_desc, uuid, template=template_,
                                         len_=len__))
    return TypeBuilder(build)


Any = AnyType('Any')
Void = Type('Void', Any)
Bool = Type('Bool', Any, ('flavour',),
            jit_checker=checkers.check_python_type([bool]),
            sampler=lambda _: bool(random.randrange(2)),
            serve_convert=lambda x: x.lower() == 'true')
Number = Type('Number', Any, ('flavour',),
              jit_checker=checkers.check_python_type([int, float]),
              serve_convert=lambda x: eval(x),
              sampler=lambda _: random.random())
Integer = Type('Integer', Number, ('flavour',),
               jit_checker=checkers.check_python_type([int]),
               serve_convert=lambda x: int(x),
               sampler=lambda _: random.randrange(10000))
Float = Type('Float', Number, ('flavour',),
             jit_checker=checkers.check_python_type([float]),
             serve_convert=lambda x: float(x),
             sampler=lambda _: random.random())
String = Type('String', Any, ('flavour', 'domain'),
              jit_checker=checkers.check_python_type([str]),
              sampler=lambda _: ''.join(random.choice(printable) for _ in range(random.randrange(100))))
Bytes = Type('Bytes', Any, ('flavour', 'domain'),
             jit_checker=checkers.check_python_type([bytes]),
             sampler=lambda _: ''.join(random.choice(printable) for _ in range(random.randrange(100))))
Bytestring = Type('Bytestring', Any, ('flavour', 'domain'),
                  jit_checker=checkers.check_python_type([type(bytes('abc', 'utf-8'))]),
                  sampler=lambda _: b''.join(random.choice(printable).encode() for _ in range(random.randrange(100))))
Dict = Type('Dict', Any, ('ktype', 'vtype', 'flavour'),
            serve_convert=lambda x: json.loads(x),
            jit_checker=checkers.check_python_type([dict]))
Set = Type('Set', Any, ('len', 'vtype', 'flavour'),
           serve_convert=lambda x: set(json.loads(x)),
           jit_checker=checkers.IterableChecker(checkers.check_python_type([set])))
Sequence = Type('Sequence', Any, ('len', 'vtype', 'flavour'),
                serve_convert=lambda x: json.loads(x),
                jit_checker=checkers.IterableChecker(checkers.check_python_type([tuple, list])))
Tuple = Type('Tuple', Sequence, ('len', 'flavour'),
             serve_convert=lambda x: tuple(json.loads(x)),
             jit_checker=checkers.IterableChecker(checkers.check_python_type([tuple])))
List = Type('List', Sequence, ('len', 'vtype', 'flavour'),
            serve_convert=lambda x: json.loads(x),
            jit_checker=checkers.IterableChecker(checkers.check_python_type([list])))


Array = Type(
    'Array',
    Any,
    ('shape', 'flavour'),
    jit_checker=checkers.ShapeChecker(numpy.ndarray),
    serve_convert=lambda x: numpy.array(json.loads(x)),
    sampler=samplers.NumpyFloatTensorSampler()
)
Tensor = Type(
    'Tensor',
    Any,
    ('shape', 'flavour'),
    jit_checker=checkers.ShapeChecker(torch.Tensor),
    serve_convert=lambda x: torch.tensor(json.loads(x)),
    sampler=samplers.FloatTensorSampler()
)
LongTensor = Type(
    'LongTensor',
    Tensor,
    ('shape', 'flavour'),
    jit_checker=checkers.ShapeChecker(torch.LongTensor, lambda x: x.to('cpu')),
    serve_convert=lambda x: torch.tensor(json.loads(x)).type(torch.long),
    sampler=samplers.LongTensorSampler()
)
FloatTensor = Type(
    'FloatTensor',
    Tensor,
    ('shape', 'flavour'),
    jit_checker=checkers.ShapeChecker(torch.FloatTensor, lambda x: x.to('cpu')),
    serve_convert=lambda x: torch.tensor(json.loads(x)).type(torch.float),
    sampler=samplers.FloatTensorSampler()
)
Image = Type('Image', Any, ('shape', 'mode', 'flavour', 'domain'),
             lambda x, y: isinstance(x, PIL.Image.Image),
             sampler=samplers.ImageSampler())

lookup = {
    'Bool': Bool,
    'Bytes': Bytes,
    'Number': Number,
    'Float': Float,
    'Array': Array,
    'Integer': Integer,
    'Tensor': Tensor,
    'FloatTensor': FloatTensor,
    'LongTensor': LongTensor,
    'Image': Image,
    'Set': Set,
    'Sequence': Sequence,
    'List': List,
    'Tuple': Tuple,
    'Dict': Dict,
    'String': String,
    'Void': Void
}
