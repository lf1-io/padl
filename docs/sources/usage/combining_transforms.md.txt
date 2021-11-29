## Combining Transforms

PADL provides functional operators, which allow to combine transforms to create powerful *compound Transforms*.

This is most useful for building deep learning pipelines on a *macro-level* - for instance combining different preprocessing steps and augmentation with a model forward pass. You can keep building the individual sub-components as you're used to - with python and PyTorch.

(compose)=

### Compose `>>`

Transforms can be **composed** using `>>`:

Composed transforms process their input in a sequence, the output of the first transform becomes the input of the second.

<img src="img/compose.png" style='display: block; margin: auto; width: 150px'>

Thus:

```python
>>> (t1 >> t2 >> t3)(x) == t3(t2(t1(x)))
True
```

### Rollout `+`

An input can be **rolled out** to multiple Transforms using `+`. This means applying different transforms to the same input. The result is a tuple.

<img src="img/rollout.png" style='display: block; margin: auto; width: 300px'>

Thus:

```python
>>> (t1 + t2 + t3)(x) == (t1(x), t2(x), t3(x))
True
```


### Parallel `/`

Multiple transform can be **applied in parallel** to multiple inputs using `/`. The input must be a tuple and the nth Transform is applied to the nth item in the tuple.

<img src="img/parallel.png" style='display: block; margin: auto; width: 300px'>

Thus:

```python
>>> (t1 / t2 / t3)((x, y, z)) == (t1(x), t2(y), t3(z))
True
```

### Map `~`

Transforms can be **mapped** using `~`. Mapping applies the same transforms to multiple inputs. The output has the same length as the input.

<img src="img/map.png" style='display: block; margin: auto; width: 300px'>

Thus:

```python
>>> (~t)([x, y, z]) == [t(x), t(y), t(z)]
True
```

### Grouping Compound Transforms

By default, compound transforms, such as rollouts and parrallels, are *flattened*. This means that even if you use parentheses to group them, the output will be a flat tuple:

```python
>>> (t1 + (t2 + t3))(x) == ((t1 + t2) + t3)(x) == (t1 + t2 + t3)(x)  == (t1(x), t2(x), t2(x))
True
```

To group them, use {meth}`padl.group`:

```python
>>> from padl import group
>>> (t1 + group(t2 + t3))(x) == (t1(x), (t2(x), t3(x)))
True
```

Continue in the {ref}`next section <stages>` to learn how to combine pre-processing, forward pass and post-processing in a single Transform.

----

### Examples

#### Compose

##### Building Pre-processing Pipelines

Use *composition* to build pre-processing pipelines - similar to as you would using `torchvision.transforms.Compose`:

```python
from padl import transform, IfTrain
from torchvision import transforms as tvt
from PIL import Image

tvt = transform(tvt)

preprocess_image = (
    transform(lambda path: Image.open(path))  # load an image from a path
    >> tvt.Resize(244, 244)  # resize the image
    >> IfTrain(tvt.RandomRotation(100))  # some augmentation
    >> tvt.PILToTensor()
)
```

This uses

- {func}`~padl.transform` to {ref}`convert a lambda function <convert-functions>` and {ref}`everything from the torchvision.transforms module <convert-module>` into a Transform
- {class}`~padl.IfTrain` to {ref}`conditionally execute a step <if-in-mode>` only during training

##### Combining Pre-processing, Model Forward Pass and Post-processing

You can use composition to combine pre-processing, model forward pass and post-processing in one transform using the special Transforms {obj}`~padl.batch` and {obj}`~padl.unbatch`:

```python
my_classifier_transform = (
    load_image                 # preprocessing ...
    >> transforms.ToTensor()   # 
    >> batch                   # ... stage
    >> models.resnet18()       # forward
    >> unbatch                 # postprocessing ...
    >> classify                # ... stage
)
```

For more details, head over to {ref}`the next section <stages>`.

#### Rollout

##### Extracting Items from a Dictionary

One common use case for the *rollout* is to extract different elements from a dictionary.

```
>>> from padl import same
>>> extract = (same['foo'] + same['bar'])
>>> extract({'foo': 1, 'baz': 2, 'bar': 3
(1, 3)
```

This uses

- {ref}`the "same" utility <same>` for getting items

(generate-image-versions)=
##### Generating Different Versions of an Image

You could also use it in a preprocessing pipeline to generate multiple views of the same image:

```python
from padl import transform, IfTrain
from torchvision import transforms as tvt
from PIL import Image

tvt = transform(tvt)
preprocess_image = (
    transform(lambda path: Image.open(path))  # load an image from a path
    >> (tvt.RandomResizedCrop(244) 
        + tvt.RandomResizedCrop(244) 
        + tvt.RandomResizedCrop(244))  # generate three different crops
)
```

This generates three different crops of the same image.

#### Parallel

##### Pass Training Samples

Use *parallel* to pass training datapoints `(input, target)` through the same pipeline:

```python

model_pass = (
    preprocess  # some preprocessing steps
    >> batch  # move to "forward" stage
    >> model  # apply PyTorch model
)

training_pipeline = (
    model_pass / batch
    >> loss  # a loss function taking a tuple (*prediction*, *target*)
)
```

This uses

- {obj}`~padl.batch` to move between {ref}`stages <stages>`.

#### Map

##### Convert Multiple Images to Tensors

To continue the {ref}`above example <generate-image-versions>`, one could use *map* to convert
all resulting `PIL Images` to tensors.

```python
from padl import transform, IfTrain
from torchvision import transforms as tvt
from PIL import Image

tvt = transform(tvt)
preprocess_image = (
    transform(lambda path: Image.open(path))  # load an image from a path
    >> (tvt.RandomResizedCrop(244) 
        + tvt.RandomResizedCrop(244) 
        + tvt.RandomResizedCrop(244))  # generate three different crops
    >> ~ tvt.PILToTensor()
)
```

This transform takes an image path and returns a tuple of tensors.
