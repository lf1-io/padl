# All you need to know about PADL

This page is a concise reference of everything you need to know about PADL.

## Any *Callable* can be a Transform

The central abstraction in PADL is the *Transform*. Essentially, a Transform
is a function.

### {meth}`~padl.transform` creates Transforms

Transforms can be created using the {func}`~padl.transform` wrapper:

```python
from padl import transform
```

{func}`~padl.transform`-decorated *functions* are Transforms:

```python
@transform
def add(x, y):  # this is a Transform
    return x + y
```

{func}`~padl.transform`-wrapped functions are Transforms:

```python
transform(lambda x: x + 1000)  # this is a Transform

from math import log10
transform(log10)  # this is a Transform
```

Instances of decorated classes implementing {meth}`__call__` are Transforms:

```python
@transform
class MinusX:
    def __init__(self, x):
        self.x = x
    
    def __call__(self, y):
        return y - self.x

[...]

minus100 = MinusX(100) #  this is a Transform
```

In particular, instances of decorated **PyTorch** Modules are Transforms:

```
@transform
class MLP(torch.nn.Module):
    def __init__(self, n_in, hidden, n_out):
        self.l1 = torch.nn.Linear(n_in, hidden)
        self.l2 = torch.nn.Linear(hidden, n_out)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        y = self.l1(x)
        y = self.relu(y)
        return self.l2(y)

[...]

mml = MLP(10, 10, 10) #  this is a Transform
```

Functions taken from wrapped modules are Transforms:

```python
import numpy as np
np = transform(np)

np.sin  # this is a transform
```

Instances of "callable" classes taken from wrapped modules are Transforms:

```python
from torch import nn
nn = transform(nn)

lin = nn.Linear(10, 10)  # this is a Transform
```


### Transforms can be combined using operators

Transforms can be *composed* using `>>`:


```python
>>> (t1 >> t2)(x) == t1(t2(x))
True
```

Transforms can be *rolled out* using `+`:

```python
>>> (t1 + t2)(x) == (t1(x), t2(x))
True
```

Transforms can be *applied in parallel* using `/`:

```python
>>> (t1 / t2)(x) == (t1(x), t2(x))
True
```

Transforms can be *mapped* using `~`:

```python
>>> (~t)([x, y, z]) == [t(x), t(y), t(z)]
True
```

Compound transforms will be *flattened*:

```python
>>> (t1 + (t2 + t3))(x) == ((t1 + t2) + t3)(x) == (t1 + t2 + t3)(x)  == (t1(x), t2(x), t2(x))
True
```

unless explicitly *grouped*:

```python
>>> from padl import group
>>> (t1 + group(t2 + t3))(x) == (t1(x), (t2(x), t3(x)))
True
```


### Special Transforms

The {obj}`~padl.identity` Transform does nothing:

```
>>> from padl import identity
>>> identity(x) == x
True
```

{obj}`~padl.batch` marks where the "forward"-part begins, {obj}`~padl.unbatch` marks where
the "postprocessing"-part begins:

```
from padl import batch, unbatch
pipeline = (
    preprocessing  # preprocessing part, single datapoints, on CPU via DataLoader
    >> batch
    >> forward  # forward part, batches, potentially on GPU
    >> unbatch
    >> postprocessing  # postprocessing part, single datapoints, on CPU
)
```

## What you can do with Transforms

### Transforms pretty-print

Transforms pretty-print in IPython:

```
>>> add
add:

   def add(x, y):
       return x + y
```

```
>>> add / identity >> add >> MinusX(100)
Compose:

      │└──────────────┐
      │               │
      ▼ (x, y)        ▼ args
   0: add           / padl.Identity()
      │
      ▼ (x, y)
   1: add          
      │
      ▼ y
   2: MinusX(x=100)
```



### Compound Transforms can be sliced

Sub-transforms of Compound Transforms can be accessed via getitem:

```
>>> (t1 >> t2 >> t3)[0] == t1
True
```

Slices work, too:

```
>>> (t1 >> t2 >> t3)[1:] == t2 >> t3
True
```

This can be used with complex, nested Transforms:

```
>>> (t1 >> t2 + t3 + t4 >> t5)[1][:2] == t2 + t3
True
```

### Applying Transforms

#### `infer_apply`
#### `eval_apply`
#### `train_apply`

### Transforms can be saved and loaded

All PADL transforms can be saved via {meth}`~padl.save`:

```
from padl import save

save(my_transform, 'mytransform.padl')
```

This creates a folder

```
mytransform.padl/
├── transform.py
└── versions.txt
```

containing a python file defining the transform and a file containing a list of all package dependencies and their versions.

When saving Transforms that are torch modules, checkpoint files with all parameters are stored, too.

Saved transforms can be loaded with {meth}`~padl.load`:

```
from padl import load

my_transform = load('mytransform.padl')
```

## Extras

### {obj}`~padl.same` is useful

The special object {obj}`~padl.same` can be used to get items from the input - 
`same[0]` is the same as `transform(lambda x: x[0])`:

```
>>> from padl import same
>>> addfirst = add >> same[0]
>>> addfirst(np.array([1, 2, 3], [2, 3, 4]))
3
```

{obj}`~padl.same` also allows to apply arbitrary methods to the input - 
`same.something()` is the same as `transform(lambda x: x.something())`:

```
>>> concat_lower = add >> same.lower()
>>> concat_lower("HELLO", "you")
"hello you"
```

### {class}`~padl.IfInfer`, {class}`~padl.IfEval` and {class}`~padl.IfTrain` apply Transforms conditionally

Use an {class}`~padl.IfInfer` Transform to apply a transform only in the "infer" stage:

```
>>> from padl import IfInfer
>>> t = MinusX(100) >> IfInfer(MinusX(100))
>>> t.infer_apply(300)
100
>>> list(t.eval_apply([300]))[0]
200
```

Analogously, use {class}`~padl.IfEval` or {class}`~padl.IfTrain` to apply a transform only in the "eval"- or "train" stage, respectively.

### {class}`~padl.Try` lets you catch exceptions

Use {class}`Try` to 

## Import anything you need `from padl`

- {func}`padl.transform` for making Transforms.
- {func}`padl.save` for saving Transforms.
- {func}`padl.load` for loading Transforms.
- {func}`padl.value` for saving by value.
- {func}`padl.group` for grouping Compound Transforms.
- {obj}`padl.identity` for doing nothing.
- {obj}`padl.batch` for defining the batchified part of a pipeline.
- {obj}`padl.unbatch` for defining the postprocessing part of a pipeline.
- {obj}`padl.same` for applying a value's method.
- {class}`padl.IfInfer` for conditioning on the 'infer'-stage.
- {class}`padl.IfEval` for conditioning on the 'eval'-stage.
- {class}`padl.IfTrain` for conditioning on the 'try'-stage.
- {class}`padl.Try` for catching exceptions in a transform.
