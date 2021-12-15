<img src="img/logo_dark_mode.png#gh-dark-mode-only" alt="PADL" width="400"><img src="img/logo.png#gh-light-mode-only" alt="PADL" width="400">

[![PyPI version](https://badge.fury.io/py/padl.svg)](https://badge.fury.io/py/padl) 
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg) 
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/padl)](https://pypi.org/project/padl/) 
[![GitHub Issues](https://img.shields.io/github/issues/lf1-io/padl.svg)](https://github.com/lf1-io/padl/issues)
[![Tests](https://github.com/lf1-io/padl/actions/workflows/actions.yml/badge.svg)](https://github.com/lf1-io/padl/actions/workflows/actions.yml)
[![codecov](https://codecov.io/gh/lf1-io/padl/branch/main/graph/badge.svg?token=NLS02IWDFQ)](https://codecov.io/gh/lf1-io/padl)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lf1-io/padl/)
[![LF1 on Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/lf1_io)

# Pipeline Abstractions for Deep Learning


Full documentation here: https://lf1-io.github.io/padl/

**PADL**:

- is a pipeline builder for **PyTorch**.
- may be used with all of the great **PyTorch** functionality you're used to for writing layers.
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
### Try out PADL in Colab notebooks
1. [MNIST](https://colab.research.google.com/github/lf1-io/padl/blob/main/notebooks/01_MNIST_using_padl.ipynb#scrollTo=bd560eb8)
2. [Simple NLP example](https://colab.research.google.com/github/lf1-io/padl/blob/main/notebooks/02_nlp_example.ipynb)
3. [Sentiment Analysis - NLP](https://colab.research.google.com/github/lf1-io/padl/blob/main/notebooks/03_Sentiment_Analysis_with_padl.ipynb)

### Resources

- Read the documentation at <https://lf1-io.github.io/padl/>.
- Find examples at <https://github.com/lf1-io/padl/tree/main/notebooks>.
- Post issues at <https://github.com/lf1-io/padl/issues>.



## Contributing

Code of conduct: https://github.com/lf1-io/padl/blob/main/CODE_OF_CONDUCT.md

If your interested in contributing to PADL please look at the current issues: https://github.com/lf1-io/padl/issues



## Licensing

PADL is licensed under the Apache License, Version 2.0. See LICENSE for the full license text.
