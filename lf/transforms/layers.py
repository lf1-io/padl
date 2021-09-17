import inspect
import torch

from lf.transforms.core import Transform
from lf.transforms.utils import get_instance_from_source, get_source
from lf.typing.types import Any
from lf.variables import StateDict

from lf.utils import evalmodel


class Layer(Transform):
    """ Transform wrapping a pytorch layer.

    :param layer: `torch.jit.ScriptModule` or `torch.nn.Module`.
    :param layer_name: name of layer.
    :param script: name of method to use (default: `forward`).
    :param _type: type tuple.
    :param in_type: input type.
    :param out_type: output type.
    :param imports: list of modules to import in order for class to work
    :param example: example to pass to class to perform jit tracing if so desired
    """

    def __init__(self, layer, layer_name, script='forward', _type=None,
                 in_type=Any(), out_type=Any(), source=None, state_dict=None,
                 imports=('torch',), example=None):

        if layer is not None and source is None and isinstance(layer, torch.nn.Module):
            source = get_source(layer)

        if state_dict is not None:
            layer.load_state_dict(state_dict)

        super().__init__(in_type=in_type, out_type=out_type,
                         script=script, layer=layer, layer_name=layer_name,
                         _type=_type, source=source, imports=imports)

        if example is not None and isinstance(self.layer, torch.nn.Module):
            self.traced(example=example)

        self.script_fn = getattr(self.layer, self.script)

    def traced(self, example=None):
        if example is None:
            example = self.type.x.sample()
        with evalmodel(self.layer):
            if not isinstance(example, tuple):
                example = example.to(self.device)
            else:
                example = tuple([ex.to(self.device) for ex in example])
            self.layer = torch.jit.trace(self.layer, example)
        return self

    def do(self, args):
        """
        :param args:
        """
        if type(args) in {tuple, list}:
            return self.script_fn(*args)
        if type(args) == torch.Tensor:
            return self.script_fn(args)
        raise TypeError('only support tensors or tuples/lists thereof')

    @property
    def layers(self):
        return {self.layer_name: self.layer}

    def train(self):
        self.layer.train()
        self._stage = 'train'

    def eval(self):
        self.layer.eval()
        self._stage = 'eval'

    def infer(self):
        self.layer.eval()
        self._stage = 'infer'

    def to_dict(self):
        """
        Convert layer to dictionaries
        :return:
        """

        if isinstance(self.layer, torch.jit.ScriptModule):
            my_dict = {
                'layer': '$' + self.layer_name,
                'script': self.script,
                'layer_name': self.layer_name,
            }
            v = {self.layer_name: self.layer}
        elif isinstance(self.layer, torch.jit._trace.TopLevelTracedModule):
            my_dict = {
                'layer': '$' + self.layer_name,
                'script': self.script,
                'layer_name': self.layer_name,
            }
            v = {self.layer_name: self.layer}
        else:
            assert isinstance(self.layer, torch.nn.Module)
            my_dict = {
                'state_dict': '$' + self.layer_name,
                'script': self.script,
                'layer_name': self.layer_name,
                'imports': self.imports,
                'source': self.source,
                'layer': {
                    'cls': self.layer.__class__.__name__,
                    'kwargs': {
                        k: getattr(self.layer, k)
                        for k in inspect.signature(self.layer.__init__).parameters
                        if k != 'self'
                    }
                },
            }
            v = {self.layer_name: StateDict(name=self.layer_name,
                                            value=self.layer.state_dict())}
        self._add_type_to_dict(my_dict)

        res = {
            'cls': self.__class__.__module__ + '.' + self.__class__.__name__,
            'kwargs': my_dict,
            'handles': self._handle_defs
        }

        if self._name is not None:
            res['_name'] = self.name
        return res, v

    @classmethod
    def from_dict(cls, d, vars=None):
        """
        Build from a dict
        :param d: dict
        :param vars: var dict
        :return:
        """
        cls.type_from_dict(d)
        layer_name = d['layer_name']
        if isinstance(vars[layer_name], torch.jit.ScriptModule) or \
                isinstance(vars[layer_name], torch.jit._trace.TopLevelTracedModule):
            t = cls(layer=vars[layer_name],
                    layer_name=d['layer_name'],
                    script=d['script'],
                    _type=d.get('_type'))
        else:
            assert (
                isinstance(vars[layer_name], dict) or
                isinstance(vars[layer_name], StateDict)
            )
            if isinstance(vars[layer_name], StateDict):
                state_dict = vars[layer_name].value
            else:
                state_dict = vars[layer_name]
            layer = get_instance_from_source(
                d['source'],
                d['imports'],
                d['layer']['cls'],
                d['layer']['kwargs'],
            )
            t = cls(layer=layer,
                    state_dict=state_dict,
                    layer_name=d['layer_name'],
                    script=d['script'],
                    source=d['source'],
                    imports=d['imports'],
                    _type=d.get('_type'))
        return t

    def repr(self):
        str_ = super().repr()[:-1] + ': "$' + self.layer_name + '/' + \
               self.script + '"'
        str_ += '\n'
        str_ += '\n'.join(['    ' + x
                           for x in self.layer.__repr__().split('\n')])
        str_ += '>'
        return str_

    def to(self, device):
        """
        Move to device
        :param device: device
        """
        self._device = device
        self.layer.to(device)
        return self

    def astr(self, with_type=True, compact=True, with_types=False):
        return self.layer_name

