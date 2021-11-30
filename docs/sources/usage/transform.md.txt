##  *Transforms* and *Pipelines*

The central abstractions in PADL are the *transform* and the *pipeline*. A transform
is a function with added functionality for the deep learning workflow. A pipeline is a combination of transforms.

With transforms and pipelines you can:

- build complex workflows with an elegant formalism and with minimal boilerplate code
- preprocess data using multi-processing without creating custom dataLoaders
- stop worrying about adding batch dimensions, dis- or enabling gradients or sending data to the GPU
- work interactively in a notebook, easily inspect, slice and debug your transforms
- save your whole workflow in a transparent, flexible format that enables reproducibility

and much more.

Read the {ref}`next section <Creating Pipelines>` to learn how to create pipelines and transforms.
