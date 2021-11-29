# Using PyTorch Modules with Transforms

Remember from {ref}`the section on creating Transforms <creating-transforms>` that PyTorch Modules can be converted into Transforms. PADL has built-in capabilities that make it convenient to work with them.

## PADL Automatically Saves PyTorch State-Dicts

When using {meth}`padl.save` to {ref}`save a Transform <saving>`, PADL automatically stores `state_dict`s of the contained PyTorch layers along with the code. When loading with {meth}`padl.load`, these are used to initialize the layers' parameters.

## Devices

Use {meth}`~padl.transforms.Transform.pd_to` to send all PyTorch layers in a transform to a device:

```
my_transform = (
    preprocess
    >> batch
    >> layer1
    >> layer2
)

my_transform.pd_to('cuda:1')
```

## Accessing Layers and Parameters

All PyTorch layers in a transform are accessible via {meth}`~padl.transforms.Transform.pd_parameters`:

```python
layers = list(my_transform.pd_layers())
```

Use {meth}`~padl.transforms.Transform.pd_parameters` to iterate over all parameters in the transform.
This can be used to initialize a PyTorch optimizer and create a training loop:

```python
predict = (
    preprocess
    >> batch
    >> layer1
    >> layer2
)
train_model = predict / batch >> loss

optimizer = torch.optim.Adam(train_model.pd_parameters(), lr=LR)

for loss in model.train_apply(TRAIN_DATA, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    o.zero_grad()
    loss.backward()
    o.step()
```

## Weight Sharing

It is possible to use the same PyTorch layer Transforms in different compound transforms. Those Transforms then share weights. This can be used, for instance, to create a train transform and an inference transform:

```
layer = MyPyTorchLayer()

train_transform = (
    load
    >> preprocess
    >> augment
    >> batch
    >> layer
    >> loss
)

infer_transform = (
    load
    >> preprocess
    >> batch
    >> layer
    >> lookup
)
```

After some training, both `train_transform` and `infer_transform` will have adapted parameters.
