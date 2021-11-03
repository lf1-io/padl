===============
Getting Started
===============

Installation
============

.. code-block:: bash

   pip install padl

Project Structure
=================

PADL's chief abstraction is ``padl.transforms.Transform``. This is an abstraction which includes all elements of a typical deep learning workflow in **PyTorch**\ :


* pre-processing
* data-loading
* batching
* forward passes in **PyTorch**
* postprocessing
* **PyTorch** loss functions

Loosely it can be thought of as a computational block with full support for **PyTorch** dynamical graphs and with the possibility to recursively combine blocks into larger blocks.

Here's an example of what this might like:

:raw-html-m2r:`<img src="img/schematic.png" width="300">`

The schematic represents a model which is a ``Transform`` instance with multiple steps and component parts; each of these are also ``Transform`` instances. The model may be applied in one pass to single data points, or to batches of data.

Basic Usage
===========

Defining atomic transforms
--------------------------

Imports:

.. code-block:: python

   from padl import this, transform, batch, unbatch, value
   import torch

Transform definition using ``transform`` decorator:

.. code-block:: python

   @transform
   def split_string(x):
       return x.split()

   @transform
   def to_tensor(x):
       x = x[:10][:]
       for _ in range(10 - len(x)):
           x.append(EOS_VALUE)
       return torch.tensor(x)

Any callable class implementing ``__call__`` can also become a transform:

.. code-block:: python

   @transform
   class ToInteger:
       def __init__(self, words):
           self.words = words + ['</s>']
           self.dictionary = dict(zip(self.words, range(len(self.words))))

       def __call__(self, word):
           if not word in self.dictionary:
               word = "<unk>"
           return self.dictionary[word]

   to_integer = ToInteger('-', ' ')

``transform`` also supports inline lambda functions as transforms:

.. code-block:: python

   split_string = transform(lambda x: x.split())

``this`` yields inline transforms which reflexively reference object methods:

.. code-block:: python

   left_shift = this[:, :-1]
   lower_case = this.lower_case()

**PyTorch** layers are first class citizens via ``padl.transforms.TorchModuleTransform``\ :

.. code-block:: python

   @transform
   class LM(torch.nn.Module):
       def __init__(self, n_words):
           super().__init__()
           self.rnn = torch.nn.GRU(64, 512, 2, batch_first=True)
           self.embed = torch.nn.Embedding(n_words, 64)
           self.project = torch.nn.Linear(512, n_words)

       def forward(self, x):
           output = self.rnn(self.embed(x))[0]
           return self.project(output)

   model = LM(N_WORDS)

   print(isinstance(layer, torch.nn.Module))                 # prints "True"
   print(isinstance(layer, padl.transforms.Transform))         # prints "True"

Finally, it's possibly to instantiate a module as a ``Transform``\ :

.. code-block:: python

   normalize = transform(torchvision).transforms.Normalize(*args, **kwargs)
   cosine = transform(numpy).cos

   print(isinstance(normalize, padl.transforms.Transform))         # prints "True"
   print(isinstance(cosine, padl.transforms.Transform))            # prints "True"

Defining compound transforms
----------------------------

Atomic transforms may be combined using 3 functional primitives:

Transform composition: **compose**

:raw-html-m2r:`<img src="img/compose.png" width="100">`

.. code-block:: python

   s = transform_1 >> transform_2

Applying a single transform over multiple inputs: **map**

:raw-html-m2r:`<img src="img/map.png" width="200">`

.. code-block:: python

   s = ~ transform

Applying transforms in parallel to multiple inputs: **parallel**

:raw-html-m2r:`<img src="img/parallel.png" width="230">`

.. code-block:: python

   s = transform_1 / transform_2

Applying multiple transforms to a single input: **rollout**

:raw-html-m2r:`<img src="img/rollout.png" width="230">`

.. code-block:: python

   s = transform_1 + transform_2

Large transforms may be built in terms of combinations of these operations. For example the branching example above would be implemented by:

.. code-block:: python

   preprocess = (
       lower_case
       >> clean
       >> tokenize
       >> ~ to_integer
       >> to_tensor
       >> batch
   )

   forward_pass = (
       left_shift
       >> IfTrain(word_dropout)
       >> model
   )

   train_model = (
       (preprocess >> model >> left_shift)
       + (preprocess >> right_shift)
   ) >> loss

Passing inputs between transform stages
---------------------------------------

In a compose model, if ``transform_1`` has 2 outputs and ``transform_2`` has 2 outputs, then in applying the composition: ``transform_1 >> transform_2`` to data, the outputs of ``transform_1`` are passed to ``transform_2`` **positionally**. So output-1 of ``transform_1`` is passed to input-1 of ``transform_2``. If ``transform_2`` has only one input, then the outputs of ``transform_1`` are passed as a tuple to ``transform_2``.

In an upcoming release, we plan to allow for passing inputs from one stage to the next using input/ output names.

Decomposing models
------------------

Often it is instructive to look at slices of a model -- this helps with e.g. checking intermediate computations:

.. code-block:: python

   preprocess[:3]

Individual components may be obtained using indexing:

.. code-block:: python

   step_1 = model[1]

Naming transforms inside models
-------------------------------

Component ``Transform`` instances may be named inline:

.. code-block:: python

   s = (transform_1 - 'a') / (transform_2 - 'b')

These components may then be referenced using ``__getitem__``\ :

.. code-block:: python

   print(s['a'] == s[0])    # prints "True"

Applying transforms to data
---------------------------

To pass single data points may be passed through the transform:

.. code-block:: python

   prediction = t.infer_apply('the cat sat on the mat .')

To pass data points in batches but no gradients:

.. code-block:: python

   for x in t.eval_apply(
       ['the cat sat on the mat', 'the dog sh...', 'the man stepped in th...', 'the man kic...'],
       batch_size=2,
       num_workers=2,
   ):
       ...

To pass data points in batches but with gradients:

.. code-block:: python

   for x in t.train_apply(
       ['the cat sat on the mat', 'the dog sh...', 'the man stepped in th...', 'the man kic...'],
       batch_size=2,
       num_workers=2,
   ):
       ...

Model training
--------------

Important methods such as all model parameters are accessible via ``Transform.pd_*``.: 

.. code-block:: python

   o = torch.optim.Adam(model.pd_parameters(), lr=LR)

For a model which emits a tensor scalar, training is super straightforward using standard torch functionality:

.. code-block:: python

   for loss in model.train_apply(TRAIN_DATA, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
       o.zero_grad()
       loss.backward()
       o.step()

Saving/ Loading
---------------

Saving:

.. code-block:: python

   model.pd_save('test.padl')

Loading:

.. code-block:: python

   from padl import load
   model = load('test.padl')

See :ref:`saving` for details.

For a full notebook example see ``notebooks/02_nlp_example.ipynb`` in the GitHub project.
