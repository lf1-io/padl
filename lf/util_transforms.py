import torch

from lf.transform import Transform


class Unbatchify(Transform):
    """Remove batch dimension (inverse of Batchify).

    :param dim: batching dimension
    """

    def __init__(self, call_info=None, lf_name=None, dim=0):
        super().__init__(call_info, lf_name=lf_name)
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
            return args.squeeze(self.dim)
        raise TypeError('only tensors and tuples of tensors recursively supported...')


class Batchify(Transform):
    """Add a batch dimension at dimension *dim*. During inference, this unsqueezes
    tensors and, recursively, tuples thereof.

    :param dim: batching dimension
    """

    def __init__(self, call_info=None, lf_name=None, dim=0):
        super().__init__(call_info, lf_name=lf_name)
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
            return args.unsqueeze(self.dim)
        if type(args) in [float, int]:
            return torch.tensor([args])
        raise TypeError('only tensors and tuples of tensors recursively supported...')
