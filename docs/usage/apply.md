(apply)=

## Applying Transforms to Data

Each Transform can be applied to data in three different **modes**: *infer*, *eval* and *train*.

To process single items in inference mode, use {meth}`.infer_apply() <padl.transforms.Transform.infer_apply>`.

```python
>>> my_classifier.infer_apply('cat.jpg')
"cat"
```

Under the hood, a batch dimension is automatically added before the *forward* {ref}`stage <stages>` and removed before the *postprocess* stage.

{meth}`.infer_apply() <padl.transforms.Transform.infer_apply>` automatically disables gradients and sends tensors to a gpu if available.

To process multiple items in eval mode, use {meth}`.eval_apply() <padl.transforms.Transform.eval_apply>`. {meth}`.eval_apply() <padl.transforms.Transform.eval_apply>` expects an iterable input and returns an output generator:

```python
>>> list(my_classifier.eval_apply(['cat.jpg', 'dog.jpg', 'airplane.jpg']))
["cat", "dog", "airplane"]
```

Internally, {meth}`.eval_apply() <padl.transforms.Transform.eval_apply>` creates a **PyTorch** {class}`~torch.utils.data.DataLoader` for the preprocessing part. This means preprocessing can be done on multiple workers.

{meth}`.eval_apply() <padl.transforms.Transform.eval_apply>` like {meth}`.infer_apply() <padl.transforms.Transform.infer_apply>`,  automatically disables gradients and sends tensors to a gpu.

To process multiple items in train mode, use {meth}`.train_apply() <padl.transforms.Transform.train_apply>`. It expects an iterable input and returns an output generator:

```python
for batch in my_classifier.pd_forward.train_apply():
    ...  # do a training update
```

{meth}`.train_apply() <padl.transforms.Transform.train_apply>` also uses a **PyTorch** {class}`~torch.utils.data.DataLoader` and, naturally, does not disable gradients.


Read the {ref}`next section <Saving and Loading>` to learn how to save and load transforms.
