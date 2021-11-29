## The *Transform*

The central abstraction in PADL is the *Transform*. Essentially, a Transform
is a function with added functionality for the deep learning workflow.

With transforms you can:

- build complex workflows with an elegant formalism and with minimal boilerplate code
- preprocess data using multi-processing without creating custom DataLoaders
- stop worrying about adding batch dimensions, dis- or enabling gradients or sending data to the GPU
- work interactively in a notebook, easily inspect, slice and debug your transforms
- save your whole workflow in a transparent, flexible format that enables reproducibility

and much more.

Read the {ref}`next section <Creating Transforms>` to learn how to create transforms.
