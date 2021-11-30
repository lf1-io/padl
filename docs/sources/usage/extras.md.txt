(extras)=

## Extras

(same)=

### The {obj}`~padl.same` Utility

The special object {obj}`~padl.same` can be used to get items from the input - 
`same[0]` is equivalent to `transform(lambda x: x[0])`:

```python
>>> from padl import same
>>> addfirst = add >> same[0]
>>> addfirst(np.array([1, 2, 3], [2, 3, 4]))
3
```

{obj}`~padl.same` also allows to apply arbitrary methods to the input - 
`same.something()` is the same as `transform(lambda x: x.something())`:

```python
>>> concat_lower = add >> same.lower()
>>> concat_lower("HELLO", "you")
"hello you"
```

(if-in-mode)=

### Applying Transforms Depending on the Mode

Often it can be useful to apply Transforms depending on what mode (infer, eval or train) is being applied. For example, you might want to apply augmentation Transforms only during training.

Use an {class}`~padl.IfInfer` Transform to apply a Transform only in the *infer* mode:

```python
>>> from padl import IfInfer
>>> t = MinusX(100) >> IfInfer(MinusX(100))
>>> t.infer_apply(300)
100
>>> list(t.eval_apply([300]))[0]
200
```

Analogously, use {class}`~padl.IfEval` or {class}`~padl.IfTrain` to apply a Transform only in the "eval"- or "train" mode, respectively.

### Exception Handling

Use {class}`~padl.Try` to handle exceptions:

```python
@transform
def open_file(filename):
    return open(filename)

@transform
def read(file):
    return file.read()

@transform
def close(file):
    file.close()
 
read_from_path = (
     open
     >> Try(read, transform(lambda x: ''), finally=close)
)
```
