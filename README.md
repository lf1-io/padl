<img src="img/logo.png" width="400">

*Transform abstractions for deep learning* -- using **Pytorch**.

## Installation

```bash
python setup.py install
```

Run tests to check:

```bash
pip install -r requirements-test.txt
pytest tests/
```

## Overview

TADL's chief abstraction is `tl.transforms.Transform`. This is an abstraction which includes all elements of a typical deep learning workflow in `pytorch`:

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
from tl import transform, batch, unbatch, group, this, transforms, importer
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

Pytorch layers are first class citizens via `tl.transforms.TorchModuleTransform`:

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
print(isinstance(layer, tl.transforms.Transform))         # prints "True"
```

Finally, it's possibly to instantiate `Transform` directly from callables using `importer`. 

```python
normalize = importer.torchvision.transforms.Normalize(*args, **kwargs)
cosine = importer.numpy.cos

print(isinstance(normalize, tl.transforms.Transform))         # prints "True"
print(isinstance(cosine, tl.transforms.Transform))            # prints "True"
```

### Defining compound transforms

Atomic transforms may be combined using 3 functional primitives:

1. Transform composition: **compose**

<img src="img/compose.png" width="100">

```python
s = transform_1 >> transform_2
```

2. Applying transforms in parallel to multiple inputs: **parallel**

<img src="img/parallel.png" width="230">

```python
s = transform_1 / transform_2
```

3. Applying multiple transforms to a single input: **rollout**

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

#### Inference mode

Single data points may be passed through the transform using `Tranform.infer_apply`:

```python
prediction = t.infer_apply('the cat sat on the mat .')
```

#### Batch modes: eval & train

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

# Licensing
TADML is licensed under the Apache License, Version 2.0. See LICENSE for the full license text.
