(saving)=
## Saving and Loading

All PADL transforms can be saved via {meth}`~padl.save`:

```
from padl import save

save(my_transform, 'mytransform.padl')
```

This creates a folder

```
mytransform.padl/
├── transform.py
└── versions.txt
```

containing a python file defining the transform and a file containing a list of all package dependencies and their versions.

When saving Transforms that are PyTorch {class}`Module`s, checkpoint files with all parameters are stored, too.

Saved transforms can be loaded with {meth}`~padl.load`:

```
from padl import load

my_transform = load('mytransform.padl')
```
