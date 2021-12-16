# Release process

Create a file `~/.pypirc` with this content:

```
[pypi]
  username = __token__
  password = <token>
```

(Token in secrets manager)

Clean up the `dist` directory:

```
rm -rf dist/
```

Make sure that the version in `padl.version.py` has been incremented in the main branch (can't be the same as any previously released version). For dev builds this would be:

```
__version__ = "x.y.zdev(w+1)"
```

Otherwise for bug-fixes it would be

```
__version__ = "x.y.(z + 1)"
```

If it's a minor release then it's:

```
__version__ = "x.(y + 1).0"
```

If it's a major release then it's:

```
__version__ = "(x + 1).0.0"
```

Once this has been done, make sure to commit and merge to `main` branch.

Then build the `pip` artifact:

```
python setup.py sdist
```

And publish the artifact to PyPi:

```
twine upload dist/*
```

Finally if that worked, and this is *not* a dev version and then tag the commit like this:

```
git tag v?.?.?
```

And push the tags:

```
git push origin v?.?.?
```

Here the `?.?.?` is the version we are currently on.