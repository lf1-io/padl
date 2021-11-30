(pytorch)=
# Using PyTorch Modules with Transforms

Remember from {ref}`the section on creating Transforms <creating-transforms>` that **PyTorch** Modules can be converted into Transforms. PADL has built-in capabilities that make it convenient to work with them.

## PADL Automatically Saves PyTorch State-Dicts

When using {meth}`padl.save` to {ref}`save a Pipeline <saving>`, PADL automatically stores `state_dict`s of the contained **PyTorch** layers along with the code. When loading with {meth}`padl.load`, these are used to initialize the layers' parameters.

## Devices

Use {meth}`~padl.transforms.Transform.pd_to` to send all **PyTorch** layers in a Transform to a device:

```python
my_pipeline = (
    preprocess
    >> batch
    >> layer1
    >> layer2
)

my_pipeline.pd_to('cuda:1')
```

## Accessing Layers and Parameters

All **PyTorch** layers in a Transform (this includes those nested within Pipelines) are accessible via {meth}`~padl.transforms.Transform.pd_layers`:

```python
layers = list(my_pipeline.pd_layers())
```

Use {meth}`~padl.transforms.Transform.pd_parameters` to iterate over all parameters in the Transform.
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

It is possible to use the same **PyTorch** layer Transforms in multiple Pipelines. Those Transforms then share weights. This can be used, for instance, to create a train Pipeline and an inference {ipeline:

```python
layer = MyPyTorchLayer()

train_pipeline = (
    load
    >> preprocess
    >> augment
    >> batch
    >> layer
    >> loss
)

infer_pipeline = (
    load
    >> preprocess
    >> batch
    >> layer
    >> lookup
)
```

After some training, both `train_pipeline` and `infer_pipeline` will have adapted parameters.

Learn about extras and utilities in the {ref}`next section <extras>`.
