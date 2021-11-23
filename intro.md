
PADL

PADL is a [minimalist, functional, unified] pytorch model builder that breaks the divide between preprocessing, forward pass and postprocessing.

Program your pytorch modules, pre- and postprocessing steps like you're used to - leveraging the full power of the python- and pytorch ecosystems - and use PADL's intuitive functional API to combine all computational steps in a single workflow.

[Forget about building data-loaders, batching, unbatching.]

Without PADL    ---     with PADL

Features:

- *everything serializes*: save your whole model (including post- and preprocessing pipelines) in a transparent and versatile format enabling reproducability and quick transition to production
- *seamless integration*: leverage the pytorch ecosystem - PADL smoothly works with *pytorch-lightning*', *torch-serve*, *torchvision*, *huggingface* (?) and many more (?).
- *notebook friendly*: pretty-print, inspect and build your models interactively and convert them into clean python modules with a simple method call
