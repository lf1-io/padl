(apply)=

## Applying Transforms to Data

Each Transform can be applied to data in three different **modes**: *infer*, *eval* and *train*.

To process single items in inference mode, use {meth}`infer_apply() <padl.transforms.Transform.infer_apply>`.

```python
>>> my_classifier.infer_apply('cat.jpg')
"cat"
```

Under the hood, a batch dimension is automatically added before the *forward* {ref}`stage <stages>` and removed before the *postprocess* stage.

{meth}`infer_apply() <padl.transforms.Transform.infer_apply>` automatically disables gradients and sends tensors to a gpu if that's set for the Pipeline (see {ref}`devices`).

To process multiple items in eval mode, use {meth}`eval_apply() <padl.transforms.Transform.eval_apply>`. {meth}`eval_apply() <padl.transforms.Transform.eval_apply>` expects an iterable input and returns an output generator:

```python
>>> list(my_classifier.eval_apply(['cat.jpg', 'dog.jpg', 'airplane.jpg']))
["cat", "dog", "airplane"]
```

Internally, {meth}`eval_apply() <padl.transforms.Transform.eval_apply>` creates a **PyTorch** {class}`~torch.utils.data.DataLoader` for the preprocessing part. All arguments available for the **PyTorch** `DataLoader` can be passed to {meth}`eval_apply() <padl.transforms.Transform.eval_apply>`, for example, preprocessing can be done with multiple workers and specified batch size with `num_workers` and `batch_size` args.

{meth}`eval_apply() <padl.transforms.Transform.eval_apply>` like {meth}`infer_apply() <padl.transforms.Transform.infer_apply>`,  automatically disables gradients and can send tensors to a gpu (see {ref}`devices`).

To process multiple items in train mode, use {meth}`train_apply() <padl.transforms.Transform.train_apply>`. It expects an iterable input and returns an output generator:

```python
for batch in my_classifier.pd_forward.train_apply():
    ...  # do a training update
```

{meth}`train_apply() <padl.transforms.Transform.train_apply>` also uses a **PyTorch** {class}`~torch.utils.data.DataLoader` for which arguments canbe passed, handles devices, and, naturally, does not disable gradients.

The outputs of {meth}`~padl.transforms.Transform.eval_apply` and
{meth}`~padl.transforms.Transform.train_apply` are {class}`~padl.transforms._GeneratorWithLength`
objects, a generator that supports {func}`len`. This allows adding progress bars, for instance using
[tqdm](https://github.com/tqdm/tqdm):

```python
from tqdm import tqdm

[...]

for batch in tqdm(my_classifier.eval_apply()):
    ...  # loop through batches showing a progress bar
```

Read the {ref}`next section <Saving and Loading>` to learn how to save and load transforms.
