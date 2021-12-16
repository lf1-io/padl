# Saving

All PADL Transforms can be saved via {py:meth}`padl.save`. For that, simply do:

```python
from padl import save

[...]

save(my_pipeline, 'some/path.padl')
```

To load a saved Transform use {py:meth}`padl.load`:

```python
from padl import load

my_pipeline = load('some/path.padl')
```

## How it works

When saving a Transform, PADL tracks down the statements that were used to create it
and writes them into a python file.

This works both with python modules and in interactive IPython sessions.

PADL will only store statements that are absolutely needed to create the Transform.
This means you can save a Transform from a messy ipython notebook and will get a clean python file.

Consider code defining the Transform {code}`mytransform`:

```python
from padl import *
import numpy as np

@transform
def plusone(x):
    return x + 1

@transform
def toarray(x):
    return np.array(x)

CONST = 100

@transform
def addconst(x):
    return x + CONST

plustwo = plusone >> plusone
mytransform = (
    toarray
    >> plustwo
    >> addconst
)

save(mytransform, 'mytransform.padl')
```

The last statement - {code}`save(..)` - creates a directory {file}`mytransform.padl` containing two files:

```
mytransform.padl/
├── transform.py
└── versions.txt
```

{file}`transform.py` contains the code needed to recreate the Transform:

```python
import numpy as np
from padl import transform


CONST = 100


@transform
def addconst(x):
    return x + CONST


@transform
def plusone(x):
    return x + 1


@transform
def toarray(x):
    return np.array(x)


_pd_main = (
    toarray
    >> plusone
    >> plusone
    >> addconst
)
```

As you can see, PADL saving supports third party imports, star imports and globally defined constants (there are some limitations to this, see {ref}`what-does-not-save`).

{file}`versions.txt` contains all the packages that were used in creating the Transforms and their versions. This is very similar to a {file}`requirements.txt` - however, it contains the package names rather that their names in the `PyPI`.

(saving-by-value)=

## Saving by value

Sometimes you might want to save the *value* of a variable instead of the python statements
that created it.

Consider the following example:

```python
from padl import transform, save

@transform
class WordIndex:
    def __init__(self, words):
        self.dictionary = {word: i for i, word in enumerate(words)}

    def __call__(self, word):
        return self.dictionary.get(word, -1)

def load_data():
    with open('my/big/textcorpus.txt') as f:
        text = f.read()
    return list(set(text.split(' ')))

words = load_data()

word_index = WordIndex(words)
save(word_index, 'word_index')
```

Here, we load a (potentially very large) dataset and process it, before using the result
to initialize a Transform.

Normally, saving {code}`word_index`, would store the python statements needed for creating the Transform, *including
the data processing* in {code}`load_data`. As loading consists in executing those statements, this would mean that the data processing would be repeated each time the Transform is loaded.

To prevent that, use {py:meth}`padl.value`:

```python
from padl import transform, save, value

[...]

word_index = WordIndex(value(words))
save(word_index, 'word_index')
```

When saving {code}`word_index`, PADL creates a json file containing the *value* of {code}`words` in the save folder which now looks like this:

```
word_index.padl/
├── 1.json
├── transform.py
└── versions.txt
```

The resulting {code}`transform.py` includes statements to load the json file:

```python
import json
from padl import transform
import pathlib


with open(pathlib.Path(__file__).parent / '1.json') as f:
    PADL_VALUE_0 = json.load(f)


@transform
class WordIndex:
    def __init__(self, words):
        self.dictionary = {word: i for i, word in enumerate(words)}
        self.not_found = len(self.dictionary)

    def __call__(self, word):
        return self.dictionary.get(word, -1)


_pd_main = WordIndex(PADL_VALUE_0)
```

Currently, this only works with values that can be serialized as JSON. Future versions of PADL will add support for other things.
You can also add your own serializer:

### Custom serializers

You can save by value using your own serializer by defining two functions, one for saving, one for loading, and passing them to
{py:meth}`padl.value`.

The *save* function expects two arguments, one for the value (*val*), one for the path (*path*):

```python
def mysaver(val, path):
    ...
```

It is responsible for saving *val* at *path*.

The *load* function expects one argument (*path*). It must load the value from there and return it.

These functions are then passed to {py:meth}`padl.value` in a tuple as the second argument. The tuple
has a third entry which defines a file suffix:

```python
x = value(val, (save, load, suffix))
```

For example, to save a value using {py:meth}`numpy.save`, you could do:

```python
def mysaver(val, path):
    np.save(path, val)

def myloader(path):
    return np.load(path)

[...]

x = value(x, (mysaver, myloader, '.npy'))

[...]
```

The *save*-function can also return one filename or multiple filenames (in a list). If it does, that return
value will be used as the path argument in the *load*-function. You can use this for more complex cases,
for instance if the value needs to be serialized in more than one file. You will not need to provide a
file suffix in this case.

For example, to save a list of numpy arrays in multiple files:

```python
def mysaver(val, path):
    filenames = []
    for i, subval in enumerate(val):
        filename = str(path) + '_{i}.npy'
        np.save(filename, val)
        filenames.append(filename)
    return filenames

def myloader(paths):
    result = []
    for path in paths:
        result.append(np.load(path))

[...]

x = value(x, (mysaver, myloader))

[...]
```

## Saving pytorch modules

When saving, PADL automatically serializes pytorch model parameters and stores them in the `.padl`
folder as `.pt` files. When loading, the parameters of the models are set to the previously saved values.

## Defining Transform within nested scopes

It is possible to create Transforms within nested scopes (for instance in a function body):

```python
from padl import transform, save

def build_my_transform(n_to_add):
    @transform
    def add_n(x):
        return x + n_to_add

    return add_n

mytransform = build_my_transform(n_to_add)
save(mytransform, 'mytransform')
```

For the saved file, PADL will undo the nesting, renaming variables in case of name conflicts.

Note that this only works one level deep, don't do something like this:

```python
def build(arg):

    def build_my_transform(n_to_add):
        @transform
        def add_n(x):
            return x + n_to_add

    return build_my_transform(arg + 100)
```

(what-does-not-save)=

## Saving Transforms from other modules

When importing Transforms from a different module, saving will by default only dump the import statement.

Thus, when doing

```python
from padl import save
from some.module import a_transform
save(a_transform, 'mytransform')
```

the `transforms.py` will contain just:

```python
from some.module import a_transform

_pd_main = a_transform
```

One can switch to a full dump of the transform using {func}`padl.fulldump`:

```python
from padl import save, fulldump
from some.module import a_transform

a_transform = fulldump(a_transform)
save(a_transform, 'mytransform')
```

This causes the `transforms.py` to contain the transform definition instead of the import:

```python
@transform
def a_transform(x):
    ...

_pd_main = a_transform
```

Instead of using {func}`padl.fulldump` on a Transform, one can apply it to a package or module. This will enable full dump for all contained transforms.

```python
from padl import save, fulldump
import some.module
from some.module import a_transform

fulldump(some.module)
```

To reverse the effect of {func}`~padl.fulldump`, use {func}`~padl.importdump`.


## What does not save

Saving PADL transforms has a few caveats you should be aware of:

### Variables defined as the targets of {code}`with` blocks

PADL currently cannot symbolically save anything that depends on the target of a with block. If you want to
use a with block in your code, either wrap it in a function or save by value ({ref}`saving-by-value`). Instead of:

```python
[...]

with open('myfile.txt') as f:
    mytransform = MyTransform(f.read())

[...]
```

do:

```python
from padl import value

[...]

def load():
    with open('myfile.txt') as f:
        return f.read()

mytransform = MyTransform(load())
[...]
```

or:

```python
from padl import value

[...]

with open('myfile.txt') as f:
    mytransform = MyTransform(value(f.read()))

[...]
```

### Variables defined as targets of loops

PADL currently cannot symbolically save anything that depends on the target of a loop:

```python
[...]

for i in range(100):
    ...

mytransform = MyTransform(i)

[...]
```

Again, save by value instead ({ref}`saving-by-value`) or wrap in a function.

### Mutated objects

PADL currently does not pick up if you mutate an object after creating it:

```python
from padl import save

[...]

a = SomeObject()
a.change()
save(MyTransform(a))
```

will not work ({code}`a.change()` will not be included in the saved {file}`transform.py`).
