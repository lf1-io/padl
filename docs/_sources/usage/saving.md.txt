(saving)=
## Saving and Loading

All PADL transforms can be saved via {meth}`~padl.save`:

```python
from padl import save

save(my_pipeline, 'mypipeline.padl')
```

This creates a folder

```python
mypipeline.padl/
├── transform.py
└── versions.txt
```

containing a python file defining the pipeline and a file containing a list of all package dependencies and their versions.

When saving pipelines which include PyTorch {class}`Module`s as transforms, checkpoint files with all parameters are stored, too.

Saved pipelines can be loaded with {meth}`~padl.load`:

```python
from padl import load

my_pipeline = load('mytransform.padl')
```
