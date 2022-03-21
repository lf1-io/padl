(saving)=
## Saving and Loading

All PADL Transforms can be saved via {meth}`~padl.save`:

```python
from padl import save

save(my_pipeline, 'mypipeline.padl')
```

This creates a folder

```python
mypipeline.padl/
├── transform.py
└── requirements.txt
```

containing a python file defining the Transform and a file with the precise requirements of the Transform.

When saving Pipelines which include PyTorch {class}`Module`s as Transforms, checkpoint files with all parameters are stored, too.

Saved Transforms can be loaded with {meth}`~padl.load`:

```python
from padl import load

my_pipeline = load('mypipeline.padl')
```

Learn in the {ref}`next section <print-slice>` how to print and slice Transforms in interactive sessions.
