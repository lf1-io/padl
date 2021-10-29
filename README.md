<img src="img/logo.png" width="400">

*Transform abstractions for deep learning* -- using **Pytorch**. 

---

Technical documentation here: https://lf1-io.github.io/tadl/

## Contents

- [Why TADL?](#why-tadl)
- [Installation](#installation)
- [Project structure](#project-structure)
- [Basic Usage](#basic-usage)
- [Licensing](#licensing)

## Why TADL?

### Problem Statement

While developing and deploying our deep learning models in **pytorch** we found that important design decisions and even data-dependent hyper-parameters took place not just in the forward passes/ modules but also in the pre-processing and post-processing. For example:

- in *NLP* the exact steps and objects necessary to convert a sentence to a tensor
- in *neural translation* the details of beam search post-processing and filtering based on business logic
- in *vision* applications, the normalization constants applied to image tensors
- in *classification* the label lookup dictionaries, formatting the tensor to human readable output

In terms of the functional mental model for deep learning we typically enjoy working with, these steps constitute key initial and end nodes on the computation graph which is executed for each model forward or backward pass.

### Standard Approach

The standard approach to deal with these steps is to maintain a library of routines for these software components and log with the model or in code which functions are necessary to deploy and use the model. This approach has several drawbacks.

- A complex versioning problem is created in which each model may require a different version of this library. This means that models using different versions cannot be served side-by-side.
- To import and use the correct pre and post processing is a laborious process when working interactively (as data scientists are accustomed to doing)
- It is difficult to create exciting variants of a model based on slightly different pre and postprocessing without first going through the steps to modify the library in a git branch or similar
- There is no easy way to robustly save and inspect the results of "quick and dirty" experimentation in, for example, jupyter notebooks. This way of operating is a major workhorse of a data-scientists' daily routine. 

### TADL Solutions

In creating **TADL** we aimed to create:

- A beautiful functional API including all mission critical computational steps in a single formalism -- pre-processing, post-processing, forward pass, batching and inference modes.
- An intuitive serialization/ saving routine, yielding nicely formatted output, saved weights and necessary data blobs which allows for easily comprehensible and reproducible results even after creating a model in a highly experimental, "notebook" fashion.
- An "interactive" or "notebook-friendly" philosophy, with print statements and model inspection designed with a view to applying and viewing the models, and inspecting model outputs.

## Installation

```bash
python setup.py install
```

Run tests to check:

```bash
pip install -r requirements-test.txt
pytest tests/
```

## Project Structure

TADL's chief abstraction is `td.transforms.Transform`. This is an abstraction which includes all elements of a typical deep learning workflow in `pytorch`:

- preprocessing
- data-loading
- batching
- forward passes in **Pytorch**
- postprocessing
- **Pytorch** loss functions

Loosely it can be thought of as a computational block with full support for **Pytorch** dynamical graphs and with the possibility to recursively combine blocks into larger blocks.

Here's a schematic of what this typically looks like:

<img src="img/schematic.png" width="300">

The schematic represents a model which is a `Transform` instance with multiple steps and component parts; each of these are also `Transform` instances. The model may be applied in one pass to single data points, or to batches of data.

## Basic Usage

### Defining atomic transforms

Imports:

```python
import tadl as td
from tadl import transform, batch, unbatch, group, this, transforms, importer
import torch
```

Transform definition using `transform` decorator:

```python
@transform
def split_string(x):
    return x.split()
    
@transform
def pad_tensor(x):
    x = x[:10]
    return torch.cat([x, torch.zeros(10 - len(x)).type(torch.long)])
    
ALPHABET = 'abcdefghijklmnopqrstuvwxyz .,-'

@transform
def lookup_letters(x):
    lookup = dict(zip(list(ALPHABET), range(len(ALPHABET))))
    return list(map(lookup.__getitem__, list(x)))
```

Any callable class implementing `__call__` can also become a transform:

```python
@transform
class Replace:
    def __init__(self, to_replace, replacement):
        self.to_replace = to_replace
        self.replacement = replacement
    def __call__(self, string):
        return string.replace(self.to_replace, self.replacement)
      
replace = Replace('-', ' ')
```

`transform` also supports inline lambda functions as transforms:

```python
split_string = transform(lambda x: x.split())
```

`this` yields inline transforms which reflexively reference object methods:

```python
index_one = this[0]
lower_case = this.lower_case()
```

Pytorch layers are first class citizens via `td.transforms.TorchModuleTransform`:

```python
@transform
class MyLayer(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.embed = torch.nn.Embedding(n_input, n_output)
    def forward(self, x):
        return self.embed(x)
      
layer = MyLayer(len(ALPHABET), 20)

print(isinstance(layer, torch.nn.Module))                 # prints "True"
print(isinstance(layer, td.transforms.Transform))         # prints "True"
```

Finally, it's possibly to instantiate `Transform` directly from callables using `importer`. 

```python
normalize = importer.torchvision.transforms.Normalize(*args, **kwargs)
cosine = importer.numpy.cos

print(isinstance(normalize, tf.transforms.Transform))         # prints "True"
print(isinstance(cosine, td.transforms.Transform))            # prints "True"
```

### Defining compound transforms

Atomic transforms may be combined using 3 functional primitives:

Transform composition: **compose**

<img src="img/compose.png" width="100">

```python
s = transform_1 >> transform_2
```

Applying transforms in parallel to multiple inputs: **parallel**

<img src="img/parallel.png" width="230">

```python
s = transform_1 / transform_2
```

Applying multiple transforms to a single input: **rollout**

<img src="img/rollout.png" width="230">

```python
s = transform_1 + transform_2
```

Large transforms may be built in terms of combinations of these operations. For example the schematic above would be implemented by:

```python
s = (
     pre_00 / pre_01
     >> pre_1
     >> pre_2
     >> batch
     >> model_1 + model_2
     >> unbatch
     >> post
)
```

Or a simple NLP string embedding model based on components defined above:

```python

model = (
    this.lower()
    >> this.strip()
    >> split_string
    >> lookup_letters
    >> transform(lambda x: torch.tensor(x))
    >> batch
    >> layer
)
```

### Decomposing models

Often it is instructive to look at slices of a model -- this helps with e.g. checking intermediate computations:

```python
preprocess = model[:4]
```

Individual components may be obtained using indexing:

```python
step_1 = model[1]
```

### Naming transforms inside models

Component `Transform` instances may be named inline:

```python
s = (transform_1 - 'a') / (transform_2 - 'b')
```

These components may then be referenced using `__getitem__`:

```python
print(s['a'] == s[0])    # prints "True"
```

### Applying transforms to data

To pass single data points may be passed through the transform:

```python
prediction = t.infer_apply('the cat sat on the mat .')
```

To pass data points in batches but no gradients:

```python
for x in t.eval_apply(
    ['the cat sat on the mat', 'the dog sh...', 'the man stepped in th...', 'the man kic...'],
    batch_size=2,
    num_workers=2,
):
    ...
```

To pass data points in batches but with gradients:

```python
for x in t.train_apply(
    ['the cat sat on the mat', 'the dog sh...', 'the man stepped in th...', 'the man kic...'],
    batch_size=2,
    num_workers=2,
):
    ...
```

### Model training

Important methods such as all model parameters are accessible via `Transform.tl_*`.: 

```python
o = torch.optim.Adam(model.tl_parameters(), lr=LR)
```

For a model which emits a tensor scalar, training is super straightforward using standard torch functionality:

```python
for loss in model.train_apply(TRAIN_DATA, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    o.zero_grad()
    loss.backward()
    o.step()
```

### NLP Example

Suppose we define a simple classifier extending our NLP pipeline:

```python
model = (
    this.lower()
    >> this.strip()
    >> split_string
    >> lookup_letters
    >> transform(lambda x: torch.tensor(x))
    >> batch
    >> layer
    >> importer.torch.nn.Linear(20, N_LABELS)
)
```

Targets to be computed are simple labels:

```python
@transform
def lookup_classes(class_):
    return next(i for i, c in enumerate(CLASSES) if c == class_)

target = (
    lookup_classes
    >> transform(lambda x: torch.tensor(x))
    >> batch
)
```

In training the model outputs can be compared with the targets with:

```python
training_pipeline = (model / target) >> loss
```

Data points must be tuples of sentences and labels.

### Weight sharing for auxiliary production models

At run-time in production we often will need important postprocessing steps on top of tensor outputs. For example, to serve meaningful predictions from our NLP model, we would want to lookup the best prediction in the `CLASSES` variable:

```python
@transform
def reverse_lookup(prediction):
    return CLASSES[prediction.topk(1)[1].item()]
```

A useful production model would be:

```python
model >> unbatch >> reverse_lookup
```

Since the weights are tied to `training_pipeline`, `model` trains together with `training_pipeline`, but with the added capability of producing human readable outputs.

## Licensing
TADL is licensed under the Apache License, Version 2.0. See LICENSE for the full license text.
