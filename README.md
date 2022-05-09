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

Just as **programs are read more often than they are written**, so are **deep learning models used more often than they are trained**.

The PyTorch ecosystem has many tools for training models. However, this is only the beginning of the journey. Once a model has been trained, it will be shared, and used in a multitude of contexts, often on a daily basis, in operations, evaluation, comparision and experimentation by data scientists. The *use* of the trained model, is how value is extracted out of its weights. Despite this important fact, support for using deep-learning models up to now has been very thin in the PyTorch ecosystem and beyond. PADL is a tool which fills this void.

**PADL**

Is great for:

- serving model predictions
- experimentation with pre-trained models in Jupyter notebooks and interactive sessions
- inspecting and visualizing intermediate model features
- evaluating and monitoring model performance on metrics and incoming test data
- computing model outputs as a preprocessing step for further data science tasks

This is how it works:

- PADL is a pipeline builder for **PyTorch**.
- may be used with all of the great **PyTorch** functionality you're used to for writing layers.
- allows users to build pre-processing, forward passes, loss functions **and** post-processing into the pipeline.
- models may have arbitrary topologies and make use of arbitrary packages from the python ecosystem.
- allows for converting standard functions to **PADL** components using a single keyword `transform`.

PADL pipelines:

- [can be trained with minimal boilerplate in PyTorch Lightning](https://medium.com/padl-developer-blog/warp-speed-model-training-in-padl-with-pytorch-lightning-645b56ba0b3a)
- [can then be served in one line with TorchServe](https://medium.com/padl-developer-blog/one-line-of-code-serves-your-pytorch-model-in-torchserve-using-padl-9e5990493181)
- interact well with the entire PyTorch ecosystem ([for example Hugging Face](https://devblog.padl.ai/build-completely-novel-pipelines-on-top-of-hugging-face-in-a-few-simple-steps-with-padl-d8da7bd8b0fe))

**PADL** was developed at [LF1](https://lf1.io/), an AI innovation lab based in Berlin, Germany.


## Getting Started

### Installation

```
pip install padl
```

PADL currently supports python **3.7**, **3.8** and **3.9**.

Python version >= **3.8** is preferred because creating and loading transforms (**not** execution) 
can be slower in **3.7**.

WARNING: PADL transforms do not work in the base Python Interpreter environment because we rely on
the inspect module to find source code (used when saving PADL transforms). Unfortunately, the 
source code typed at this interactive prompt is discarded as soon as it is parsed. 
Therefore, we recommend using the IPython interpreter or Jupyter Notebooks for interactive sessions.

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
1. [Basic PADL](https://colab.research.google.com/github/lf1-io/padl/blob/main/notebooks/00_basic_padl.ipynb)
1. [MNIST](https://colab.research.google.com/github/lf1-io/padl/blob/main/notebooks/01_MNIST_using_padl.ipynb)
1. [Simple NLP example](https://colab.research.google.com/github/lf1-io/padl/blob/main/notebooks/02_nlp_example.ipynb)
1. [Sentiment Analysis - NLP](https://colab.research.google.com/github/lf1-io/padl/blob/main/notebooks/03_Sentiment_Analysis_with_padl.ipynb)
1. [DC-GAN - Image Generation](https://colab.research.google.com/github/lf1-io/padl/blob/main/notebooks/04_DCGAN.ipynb)
1. [CLIP guided diffusion for face editing](https://colab.research.google.com/github/lf1-io/padl/blob/main/notebooks/05_diffuse_faces.ipynb)


### Resources

- Read the documentation at <https://lf1-io.github.io/padl/>.
- Find examples at <https://github.com/lf1-io/padl/tree/main/notebooks>.
- Post issues at <https://github.com/lf1-io/padl/issues>.


## Dev Blog
Read more about PADL on the [PADL developer's blog](https://devblog.padl.ai/)

## Contributing

[Code of conduct](https://github.com/lf1-io/padl/blob/main/CODE_OF_CONDUCT.md)

If you're interested in contributing to PADL please look at the [current issues](https://github.com/lf1-io/padl/issues)


## Licensing

PADL is licensed under the Apache License, Version 2.0. See LICENSE for the full license text.
