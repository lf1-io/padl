======
Saving
======

All PADL transforms can be saved via :py:meth:`padl.save`. For that, simply do::

    from padl import save

    [...]

    save(my_transform, 'some/path.padl')

To load a saved transform use :py:meth:`padl.load`::

    from padl import load

    my_transform = load('some/path.padl')


How it works
============

When saving a transform, PADL tracks down the statements that were used to create it
and writes them into a python file.

This works both with python modules and in interactive IPython sessions.

PADL will only store statements that are absolutely needed to create the transform.
This means you can save a transform from a messy ipython notebook and will get a clean python file.

Consider code defining the transform :code:`mytransform`::

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
        return x * CONST

    plustwo = plusone >> plusone
    mytransform = (
        toarray
        >> plustwo
        >> addconst
    )

    save(mytransform, 'mytransform.padl')

The last statement - :code:`save(..)` - creates a directory :file:`mytransform.padl` containing two files::

    mytransform.padl/
    ├── transform.py
    └── versions.txt


:file:`transform.py` contains the code needed to recreate the transform::

    import numpy as np
    from padl import transform


    CONST = 100


    @transform
    def addconst(x):
        return x * CONST


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


As you can see, PADL saving supports third party imports, star imports and globally defined constants (there are some limitations to this, see :ref:`what-does-not-save`).

:file:`versions.txt` contains all the packages that were used in creating the transforms and their versions. This is very similar to a :file:`requirements.txt` - however, it contains the package names rather that their names in the `PyPI`.


.. _saving-by-value:


Saving by value
===============

Sometimes you might want to save the *value* of a variable instead of the python statements
that created it.

Consider the following example::
    
    from padl import transform, save

    @transform
    class WordIndex:
        def __init__(self, words):
            self.dictionary = {word: i for i, word in enumerate(words)}
            self.not_found = len(self.dictionary)

        def __call__(self, word):
            return self.dictionary.get(word, -1)

    def load_data():
        with open('my/big/textcorpus.txt') as f:
            text = f.read()
        list(set(text.split(' ')))

    words = load_data()

    word_index = WordIndex(words)
    save(word_index, 'word_index')


Here, we load a (potentially very large) dataset and process it, before using the result
to initialize a transform.

Normally, saving :code:`word_index`, would store the python statements needed for creating the transform, *including 
the data processing* in :code:`load_data`. As loading consists in executing those statements, this would mean that the data processing would be repeated each time the transform is loaded.

To prevent that, use :py:meth:`padl.value`::

    from padl import transform, save, value

    [...]

    word_index = WordIndex(value(words))
    save(word_index, 'word_index')

When saving :code:`word_index`, PADL creates a json file containing the *value* of :code:`words` in the save folder which now looks like this::

    word_index.padl/
    ├── 1.json
    ├── transform.py
    └── versions.txt

The resulting :code:`transform.py` includes statements to load the json file::

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

Currently, this only works with values that can be serialized as JSON. Future versions of PADL will add support for other things.


Saving pytorch modules
======================

When saving, PADL automatically serializes pytorch model parameters and stores them in the `.padl`
folder as `.pt` files. When loading, the parameters of the models are set to the previously saved values.


Defining transform within nested scopes
=======================================

It is possible to create transforms within nested scopes (for instance in a function body)::

    from padl import transform, save

    def build_my_transform(n_to_add):
        @transform
        def add_n(x):
            return x + n_to_add

        return add_n

    mytransform = build_my_transform(n_to_add)
    save(mytransform, 'mytransform')


For the saved file, PADL will undo the nesting, renaming variables in case of name conflicts.

Note that this only works one level deep, don't do something like this::

    def build(arg):

        def build_my_transform(n_to_add):
            @transform
            def add_n(x):
                return x + n_to_add

        return build_my_transform(arg + 100)


.. _what-does-not-save:

What does not save
==================

Saving PADL transforms has a few caveats you should be aware of:


Variables defined as the targets of :code:`with` blocks
-------------------------------------------------------

PADL currently cannot symbolically save anything that depends on the target of a with block. If you want to 
use a with block in your code, either wrap it in a function or save by value (:ref:`saving-by-value`). Instead of::

    [...]

    with open('myfile.txt') as f:
        mytransform = MyTransform(f.read())

    [...]
 
do::

    from padl import value

    [...]

    def load():
        with open('myfile.txt') as f:
            return f.read()

    mytransform = MyTransform(load())
    [...]

or::

    from padl import value

    [...]

    with open('myfile.txt') as f:
        mytransform = MyTransform(value(f.read()))

    [...]


Variables defined as targets of loops
-------------------------------------

PADL currently cannot symbolically save anything that depends on the target of a loop::

    [...]

    for i in range(100):
        ...

    mytransform = MyTransform(i)

    [...]

Again, save by value instead (:ref:`saving-by-value`) or wrap in a function.

Mutated objects
---------------

PADL currently does not pick up if you mutate an object after creating it::

    from padl import save

    [...]

    a = SomeObject()
    a.change()
    save(MyTransform(a))

will not work (:code:`a.change()` will not be included in the saved :file:`transform.py`).
