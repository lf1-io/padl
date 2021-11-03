.. role:: raw-html-m2r(raw)
   :format: html

============
Introduction
============

**PyTorch** *abstractions for deep learning*.

----

Full documentation here: https://lf1-io.github.io/padl/

**PADL**\ :


* is a model builder for **PyTorch**. Build models with a functional API featuring operator overloading. Super fun and easy to use. Use **PADL** together with all of the great functionality you're used to with **Pytorch** for saving, and writing layers.
* allows users to build pre-processing, forward passes, loss functions **and** post-processing into the model
* models may have arbitrary topologies and make use of arbitrary packages from the python ecosystem
* allows for converting standard functions to **PADL** components using a single keyword ``transform``.

**PADL** was developed at `LF1 <https://lf1.io/>`_ an AI innovation lab based in Berlin, Germany.

Why PADL?
---------

For data scientists, developing neural network models is often hard to coordinate and manage, due to the need to juggle diverse tasks such as pre-processing, **PyTorch** layers, loss functions and post-processing, as well as maintenance of config files, code bases and communicating results between teams. PADL is a tool to alleviate several aspects of this work.

Problem Statement
^^^^^^^^^^^^^^^^^

While developing and deploying our deep learning models in **PyTorch**\ , we found that important design decisions and even data-dependent hyper-parameters took place not just in the forward passes/ modules but also in the pre-processing and post-processing. For example:


* in *NLP* the exact steps and objects necessary to convert a sentence to a tensor
* in *neural translation* the details of beam search post-processing and filtering based on business logic
* in *vision* applications, the normalization constants applied to image tensors
* in *classification* the label lookup dictionaries, formatting the tensor to human readable output

In terms of the functional mental model for deep learning we typically enjoy working with, these steps constitute key initial and end nodes on the computation graph which is executed for each model forward or backward pass.

Standard Approach
^^^^^^^^^^^^^^^^^

The standard approach to deal with these steps is to maintain a library of routines for these software components and log with the model or in code which functions are necessary to deploy and use the model. This approach has several drawbacks.


* A complex versioning problem is created in which each model may require a different version of this library. This means that models using different versions cannot be served side-by-side.
* To import and use the correct pre- and post-processing is a laborious process when working interactively (as data scientists are accustomed to doing)
* It is difficult to create exciting variants of a model based on slightly different pre- and post-processing without first going through the steps to modify the library in a git branch or similar
* There is no easy way to robustly save and inspect the results of "quick and dirty" experimentation in, for example, jupyter notebooks. This way of operating is a major workhorse of a data-scientists' daily routine. 

PADL Solutions
^^^^^^^^^^^^^^

In creating **PADL** we aimed to create:


* A beautiful functional API including all mission critical computational steps in a single formalism -- pre-processing, post-processing, forward pass, batching and inference modes.
* An intuitive serialization/ saving routine, yielding nicely formatted output, saved weights and necessary data blobs which allows for easily comprehensible and reproducible results even after creating a model in a highly experimental, "notebook" fashion.
* An "interactive" or "notebook-friendly" philosophy, with print statements and model inspection designed with a view to applying and viewing the models, and inspecting model outputs.

With **PADL** it's easy to maintain a single pipeline object for each experiment which includes pre-processing, forward pass and post-processing, based on the central ``Transform`` abstraction. When the time comes to inspect previous results, simply load that object and inspect the model topology and outputs interactively in a **Jupyter** or **IPython** session. When moving to production, simply load the entire pipeline into the serving environment or app, without needing to maintain disparate libraries for the various model components. If the experiment needs to be reproduced down the line, then simply re-execute the experiment by pointing the training function to the saved model output. 


License
-------

PADL is licensed under the Apache License, Version 2.0. See LICENSE for the full license text.
