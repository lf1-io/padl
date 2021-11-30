(creating-transforms)=
## Creating Transforms

Everything that can be *called* in python can be converted into a {class}`~padl.transforms.Transform`.
For that, use the {func}`~padl.transform` wrapper.

```python
from padl import transform
```

(convert-functions)=

**Functions** can be converted to transforms:

- *{func}`~padl.transform`-decorated functions* are Transforms

    ```python
    @transform
    def add(x, y):  # this is a Transform
        return x + y
    ```

- *{func}`~padl.transform`-wrapped functions* are Transforms

    ```python
    transform(lambda x: x + 1000)  # this is a Transform

    from math import log10
    transform(log10)  # this is a Transform
    ```

**Instances of classes** can become transforms, too:

- *Instances of {func}`~padl.transform`- decorated classes implementing {meth}`__call__`* are Transforms

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

- In particular, *instances of decorated **PyTorch** {class}`Module`s* are Transforms

    ```python
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

(convert-module)=

It's also possible to wrap **entire python modules**. When doing this, everything that comes out of the module is a Transform:

- *Functions taken from wrapped modules* are Transforms

    ```python
    import numpy as np
    np = transform(np)

    np.sin  # this is a transform
    ```

- *Instances of "callable" classes taken from wrapped modules* are Transforms

    ```python
    from torch import nn
    nn = transform(nn)

    lin = nn.Linear(10, 10)  # this is a Transform
    ```

Learn in the {ref}`next Section <pipelines>` how to combine multiple Transforms to form a Pipeline.

### Examples
