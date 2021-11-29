(stages)=

## Stages: Preprocess, Forward and Postprocess

Each Transform has a *preprocess*, *forward* and *postprocess* part. We call those parts **stages**.

As the names suggest, the different stages are responsible for processing data in the different parts of the deep learning workflow:

- *preprocess* stands for pre-processing - for example: loading, reshaping and augmenting data
- *forward* corresponds to the model's forward part - what happens in a PyTorch module, usually on the gpu
- *postprocess* stands for post-processing - for example converting the output of a model to a readable format

To define stages, use the special Transforms {obj}`padl.batch` and
{obj}`padl.unbatch` in a {ref}`composed <compose>` Transform:

```{code-block} python
:emphasize-lines: 19, 21

from padl import transform, batch, unbatch
from torchvision import transforms, models

transforms = transform(tansforms)
models = transform(models)

@transform
def load_image(path):
    return Image.load()

@transform
def classify(x):
    # [...] lookup the most likely class
    return class

my_classifier_transform = (
    load_image                 # preprocessing ...
    >> transforms.ToTensor()   # 
    >> batch                   # ... stage
    >> models.resnet18()       # forward
    >> unbatch                 # postprocessing ...
    >> classify                # ... stage
)
```

The different stages of a transform can be accessed via {meth}`Tranfroms.pd_preprocess <padl.transforms.Transform.pd_preprocess>`, {meth}`Transform.pd_forward <padl.transforms.Transform.pd_forward>` and {meth}`Transform.pd_postprocess <padl.transforms.Transform.pd_postprocess>`:

```python
>>> my_classifier_transform.pd_preprocess
load_image >> transforms.ToTensor() >> batch
>>> my_classifier_transform.pd_forwad
models.resnet18()
>>> my_classifier_transform.pd_postprocess
unbatch >> classify
```

The Transforms in the preprocess and postprocess stages process single *items* whereas the transforms in the forward stage process *batches*.

Continue in the {ref}`next section <apply>` to learn how to apply transforms to data for inference, evaluation and training.
