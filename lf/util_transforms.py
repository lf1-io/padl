import torch

from lf.transform import ClassTransform


class Unbatchify(ClassTransform):
    """Mark start of postprocessing

    Unbatchify removes batch dimension (inverse of Batchify) and moves the input tensors to 'cpu'.

    :param dim: batching dimension
    """

    def __init__(self, dim=0, lf_name=None):
        super().__init__(lf_name=lf_name)
        self.dim = dim
        self._lf_component = {'postprocess'}

    def __call__(self, args):
        assert self.lf_stage is not None,\
            'Stage is not set, use infer_apply, eval_apply or train_apply'

        if self.lf_stage != 'infer':
            return args
        if isinstance(args, tuple):
            return tuple([self(x) for x in args])
        if isinstance(args, torch.Tensor):
            return args.squeeze(self.dim).to('cpu')

        raise TypeError('only tensors and tuples of tensors recursively supported...')


class Batchify(ClassTransform):
    """Mark end of preprocessing.

    Bachify adds batch dimension at *dim*. During inference, this unsqueezes tensors and,
    recursively, tuples thereof. Batchify also moves the input tensors to device specified
    for the transform.

    :param dim: batching dimension
    """

    def __init__(self, dim=0, lf_name=None):
        super().__init__(lf_name=lf_name)
        self.dim = dim
        self._lf_component = {'preprocess'}

    def __call__(self, args):
        assert self.lf_stage is not None,\
            'Stage is not set, use infer_apply, eval_apply or train_apply'

        if self.lf_stage != 'infer':
            return args
        if type(args) in {tuple, list}:
            return tuple([self(x) for x in args])
        if isinstance(args, torch.Tensor):
            return args.unsqueeze(self.dim).to(self.lf_device)
        if type(args) in [float, int]:
            return torch.tensor([args]).to(self.lf_device)
        raise TypeError('only tensors and tuples of tensors recursively supported...')
