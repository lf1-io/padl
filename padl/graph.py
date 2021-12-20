import networkx as nx
import padl
from IPython.display import Image
from copy import copy, deepcopy


class Node:
    _id = 0

    def __init__(self, transform=None, name=None):
        self.id = self.generate_id()
        self.transform = transform
        self.name = name
        self.in_node = []
        self.out_node = []
        self.out_index = None
        self._input_args = None
        self._output_args = {}
        self.pd_output = padl.transforms._OutputSlicer(self)#self.transform._pd_output_slice
        self.set_output_slice()

    @property
    def name(self):
        return self._name

    @property
    def name_id(self):
        return self._name_id

    @name.setter
    def name(self, new_name):
        self._name = new_name
        if self._name is None and self.transform is not None:
            self._name = self.transform.pd_name
        if self._name is not None:
            self._name_id =self._name + ' ' + str(self.id)
        else:
            self._name_id = str(self.id)

    def set_output_slice(self):
        if isinstance(self.transform, padl.transforms.Transform):
            self._pd_output_slice = self.transform._pd_output_slice
            self.transform._pd_output_slice = None
            return
        self._pd_output_slice = None

    def get_output_slice(self):
        out = self._pd_output_slice
        self._pd_output_slice = None
        return out

    @classmethod
    def generate_id(cls):
        cls._id += 1
        return cls._id

    def __call__(self, args):
        # self._output = self.transform(self.args)
        # if self.out_index is not None:
        #     return self.output_node[self.out_index]
        return self.pd_call_node(args)

    def __hash__(self):
        return hash(repr(self))

    def __deepcopy__(self, memo):
        """Deepcopy of Nodes"""
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                self.transform,
                self.name)
            _copy.in_node = copy(self.in_node)
            _copy.out_node = copy(self.out_node)
            memo[id_self] = _copy
        return _copy

    def connect_innode(self, node):
        if node not in self.in_node:
            self.in_node.append(node)
            node.connect_outnode(self)
            self._create_input_args_dict()

    def connect_outnode(self, node):
        if node not in self.out_node:
            self.out_node.append(node)
            node.connect_innode(self)
            self._output_args[node] = self.get_output_slice()

    def _create_input_args_dict(self):
        """Update dictionary of args for input"""
        self._input_args = {node: {'args': None, 'updated': False} for node in self.in_node}
        if len(self._input_args) == 0:
            self._input_args[None] = {'args': None, 'updated': False}

    def _register_input_args(self, args, in_node):
        """Register the input args"""
        assert in_node in self._input_args.keys(), f"{in_node.name} not connected to {self.name}"
        self._input_args[in_node]['args'] = args
        self._input_args[in_node]['updated'] = True

        if all([in_args['updated'] for _, in_args in self._input_args.items()]):
            return True
        return False

    def pd_call_transform(self, args):
        return self.transform.pd_call_transform(args)

    def gather_args(self):
        args = [in_args['args'] for _, in_args in self._input_args.items()]
        if len(args) == 1:
            return args[0]
        return tuple(args)

    def _update_args(self, args, out_node):
        slice = self._output_args[out_node]
        if slice is not None:
            return args[slice]
        return args

    def pd_call_node(self, args, in_node=None):
        if self._input_args is None:
            self._create_input_args_dict()
        if self._register_input_args(args, in_node):
            args = self.gather_args()
            output = None
            args_to_pass = self.pd_call_transform(args)
            for out_node in self.out_node:
                updated_arg = self._update_args(args_to_pass, out_node)
                output = out_node.pd_call_node(updated_arg, self)
            return output if output is not None else args_to_pass


class Graph(Node):
    def __init__(self, transforms=None, connection=None):
        super().__init__()
        self.transforms = transforms
        self.input_node = Node(padl.Identity(), name='Input')
        self.output_node = Node(padl.Identity(), name='Output')
        self._input_args = None
        self.networkx_graph = None
        self.nodes = []

    def _store_transform_nodes(self, transforms):
        for t_ in transforms:
            if isinstance(t_, Node):
                self.nodes.append(deepcopy(t_))
                #self.nodes.append(t_)
            else:
                self.nodes.append(Node(t_))

    def connect_innode(self, node):
        if node not in self.input_node.in_node:
            self.input_node.in_node.append(node)
            node.connect_outnode(self.input_node)
            self.input_node._create_input_args_dict()

    def connect_outnode(self, node):
        # import pdb; pdb.set_trace()
        if node not in self.output_node.out_node:
            # self.output_node.out_node.append(node)
            self.output_node.connect_outnode(node)
            node.connect_innode(self.output_node)
            self.output_node._output_args[node] = self.get_output_slice()

    def compose(self, transforms):
        self._store_transform_nodes(transforms)

        node = self.nodes[0]
        node.connect_innode(self.input_node)

        for next_node in self.nodes[1:]:
            # import pdb;pdb.set_trace();
            next_node.connect_innode(node)
            node = next_node
        # import pdb; pdb.set_trace();
        next_node.connect_outnode(self.output_node)

    def rollout(self, transforms):
        self._store_transform_nodes(transforms)

        for node in self.nodes:
            node.connect_innode(self.input_node)
            node.connect_outnode(self.output_node)

    def parallel(self, transforms):
        self._store_transform_nodes(transforms)

        for indx, node in enumerate(self.nodes):
            node.connect_innode(self.input_node.pd_output[indx])
            node.connect_outnode(self.output_node)

    def pd_call_node(self, args, in_node=None):
        output = self.input_node.pd_call_node(args, in_node)

        return output

    def __hash__(self):
        return hash(repr(self))

    def __deepcopy__(self, memo):
        """Deepcopy of Nodes"""
        def _copy_nodes(_copy_graph, inp_node, copy_inp_node):
            for node in inp_node.out_node:
                copy_node = deepcopy(node)
                _copy_graph.nodes.append(copy_node)

                copy_node.in_node = []
                copy_node.out_node = []

                copy_inp_node.connect_outnode(copy_node)
                _copy_nodes(_copy_graph, node, copy_node)

            if len(inp_node.out_node) == 0:
                _copy_graph.output_node = copy_inp_node

        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                self.transform)
            _copy.in_node = copy(self.in_node)
            _copy.out_node = copy(self.out_node)

            _copy.input_node = deepcopy(self.input_node)
            _copy.input_node.in_node = []
            _copy.input_node.out_node = []
            _copy_nodes(_copy, self.input_node, _copy.input_node)

            memo[id_self] = _copy
        return _copy

    """
    def parallel(self, transforms):
        self._store_transform_nodes(transforms)
        
        for ind, node in enumerate(self.nodes):
    """

    def _add_to_networkx_graph_id(self, innode, networkx_graph):
        for node in innode.out_node:
            networkx_graph.add_node(node.id, node=node)
            networkx_graph.add_edge(innode.id, node.id)
            self._add_to_networkx_graph_id(node, networkx_graph)

    def _add_to_networkx_graph_name(self, innode, networkx_graph):
        for node in innode.out_node:
            networkx_graph.add_node(node.name_id, node=node)
            networkx_graph.add_edge(innode.name_id, node.name_id)
            self._add_to_networkx_graph_name(node, networkx_graph)

    def convert_to_networkx(self, with_name=False):
        node = self.input_node
        self.networkx_graph = nx.DiGraph()

        if with_name:
            self.networkx_graph.add_node(node.name_id, node=node)
            self._add_to_networkx_graph_name(node, self.networkx_graph)
            return self.networkx_graph

        self.networkx_graph.add_node(node.id, node=node)
        self._add_to_networkx_graph_id(node, self.networkx_graph)
        return self.networkx_graph

    def draw_nx(self, with_name=True, with_labels=True, layout='spring_layout', **kwargs):
        self.convert_to_networkx(with_name=with_name)
        inbuilt_kwargs = dict(
            with_labels=with_labels,
            node_size=1400,
            node_shape="o",
            width=1,
            alpha=0.9,
            linewidths=3,
            node_color='lightblue',
        )
        inbuilt_kwargs.update(kwargs)
        #layout_func = getattr(nx.drawing.layout, layout)
        pos = nx.nx_agraph.graphviz_layout(self.networkx_graph, prog='dot')
        #pos = layout_func(self.networkx_graph)
        return nx.draw(self.networkx_graph, pos, **inbuilt_kwargs)

    def draw(self, with_name=True):
        self.convert_to_networkx(with_name=with_name)
        dot = nx.nx_agraph.to_agraph(self.networkx_graph)
        dot.layout('dot')
        return Image(dot.draw(format='png', prog='dot'))

    def count_batchify(self):
        def _count_batchify(node, counter=0):
            if isinstance(node, Graph):
                out_nodes = node.input_node.out_node
            else:
                out_nodes = node.out_node

            for out_node in out_nodes:
                if isinstance(out_node.transform, padl.Batchify):
                    counter += 1
                assert counter < 2, f'Error: more than 1 Batchify in single path till {out_node.name}'
                counter = _count_batchify(out_node, counter)

            return counter

        counter = _count_batchify(self.input_node)
        return counter
