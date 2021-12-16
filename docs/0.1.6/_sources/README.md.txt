```{eval-rst}
.. role:: raw-html-m2r(raw)
   :format: html
```

# Introduction

**Pipeline abstractions for deep learning**

---

**PADL**:

- is a pipeline builder for **PyTorch**. 
- can be used with all of the great **PyTorch** functionality you're used to for writing layers.
- allows users to build pre-processing, forward passes, loss functions **and** post-processing into the pipeline.
- models may have arbitrary topologies and make use of arbitrary packages from the python ecosystem.
- allows for converting standard functions to **PADL** components using a single keyword `transform`.

**PADL** was developed at [LF1](https://lf1.io/), an AI innovation lab based in Berlin, Germany.

## Getting Started

### Installation

```
pip install padl
```

PADL currently supports python **3.7**, **3.8** and **3.9**.

Python version >= **3.8** is preferred because creating and loading transforms (**not** execution) 
can be slower in **3.7**.

### Your first PADL program

```python
from padl import transform, batch, unbatch
import torch
from torch import nn
nn = transform(nn)

@transform
def prepare(x):
    return torch.tensor(x)

@transform
def post(x):
    return x.topk(1)[1].item()

my_pipeline = prepare >> batch >> nn.Linear(10, 20) >> unbatch >> post
```

### Resources

- Go to {ref}`usage` for a concise but comprehensive manual.
- Find examples at <https://github.com/lf1-io/padl/tree/main/notebooks>.
- Post issues at <https://github.com/lf1-io/padl/issues>.

## Contributing

Code of conduct: <https://github.com/lf1-io/padl/blob/main/CODE_OF_CONDUCT.md>.

If your interested in contributing to PADL please look at the current issues: <https://github.com/lf1-io/padl/issues>.

## Licensing

PADL is licensed under the Apache License, Version 2.0. See LICENSE for the full license text.
