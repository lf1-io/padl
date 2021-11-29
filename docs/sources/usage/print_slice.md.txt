## Printing and Slicing

Pipeline pretty-print in IPython:

```python
>>> add
add:

   def add(x, y):
       return x + y
```

```python
>>> add / identity >> add >> MinusX(100)
Compose:

      │└──────────────┐
      │               │
      ▼ (x, y)        ▼ args
   0: add           / padl.Identity()
      │
      ▼ (x, y)
   1: add          
      │
      ▼ y
   2: MinusX(x=100)
```

Sub-pipelines of pipelines can be accessed via getitem:

```python
>>> (t1 >> t2 >> t3)[0] == t1
True
```

Slices work, too:

```python
>>> (t1 >> t2 >> t3)[1:] == t2 >> t3
True
```

This can be used with complex, nested pipelines:

```python
>>> (t1 >> t2 + t3 + t4 >> t5)[1][:2] == t2 + t3
True
```
