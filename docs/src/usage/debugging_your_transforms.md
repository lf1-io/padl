(debugging-your-transforms)=
## Debugging your Transforms

PADL provides a customized debugger to inspect and diagnose errors raised when applying a 
{class}`~padl.transforms.Pipeline`.

For using this functionality, call {obj}`~pd_debug` after the execution failed, such as 

```python
from padl import pd_debug

pd_debug()
```

This way, we get a debugger where we can interact with the different parts of the 
{class}`~padl.transforms.Pipeline` and inspect relevant information on each level.

The debugger console has defined the following commands:
```
* u(p): go one abstraction level up on the {class}`~padl.transforms.Pipeline`.
* d(own): go one abstraction level down on the {class}`~padl.transforms.Pipeline`.
* w(here am I?): show code position where the {class}`~padl.transforms.Transform` was defined.
* i(nput): show input to this level.
* h(elp): print help about the commands.
* q(uit): quit the debugger.
```

On top of this, it incorporates an in-line compiler so the user can define and inspect variables, 
check attributes, import packages...

#### Example

Let {obj}`~t` be the following {class}`~padl.transforms.Pipeline`.

```python
import torch
from padl import transform, batch

to_tensor = transform(lambda x: torch.LongTensor(x))
emb = transform(torch.nn.Embedding)(10, 8)
linear =  transform(torch.nn.Linear)(5, 5)
t = to_tensor >> batch >> emb >> linear
```

Then, if we run

```python
list(t.train_apply([[9,8,8], [4, 4, 4], [5, 5, 5], [6, 6, 6]], batch_size=2, num_workers=0))
```

we get an error since the shape of the embeddings does not match wit the one expected by the linear
layer. Calling {obj}`~pd_debug` after this will retrieve a console where we can make some checks. 

At the starting level of the debugger, we interact with the entire Transform and its absolute 
input. One level down, it goes directly to the **stage** that got the Exception (*preprocess*, 
*forward* or *postprocess*), and each level deeper moves recursively inside the element that failed
until the {class}`~padl.transforms.AtomicTransform` that failed is reached.

Below, some examples are provided:

```python
> i

[[9, 8, 8], [4, 4, 4]]

> t

Compose - "t":

      │
      ▼ x
   0: to_tensor                                    
      │
      ▼ args
   1: Batchify(dim=0)                              
      │
      ▼ input
   2: Embedding(num_embeddings=10, embedding_dim=8)
      │
      ▼ input
   3: Linear(-?-)                                    <---- error here 

>  d

Compose:

      │
      ▼ input
   0: Embedding(num_embeddings=10, embedding_dim=8)
      │
      ▼ input
   1: Linear(-?-)                                    <---- error here 

> d

Linear(-?-) - "linear":

   Linear(in_features=5, out_features=5, bias=True)  <---- error here 

> i.shape

torch.Size([2, 3, 8])

> a = 4
> a

4

> q
```

