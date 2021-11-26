# Usage

The following collection of pages are a concise reference of everything you need to know to use PADL.
Read them all to know everything you need to get started!


```{toctree}
:caption: 'Contents:'
:maxdepth: 1

usage/transform.md
usage/creating_transforms.md
usage/combining_transforms.md
usage/stages.md
usage/apply.md
usage/saving.md
usage/print_slice.md
usage/pytorch.md
usage/extras.md
```


## Import anything you need `from padl`

- {func}`~padl.transform` for making Transforms.
- {func}`~padl.save` for saving Transforms.
- {func}`~padl.load` for loading Transforms.
- {func}`~padl.value` for saving by value.
- {func}`~padl.group` for grouping Compound Transforms.
- {obj}`~padl.identity` for doing nothing.
- {obj}`~padl.batch` for defining the batchified part of a pipeline.
- {obj}`~padl.unbatch` for defining the postprocessing part of a pipeline.
- {obj}`~padl.same` for applying a value's method.
- {class}`~padl.IfInfer` for conditioning on the 'infer'-stage.
- {class}`~padl.IfEval` for conditioning on the 'eval'-stage.
- {class}`~padl.IfTrain` for conditioning on the 'try'-stage.
- {class}`~padl.Try` for catching exceptions in a transform.
