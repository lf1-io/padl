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
        self._output_slice = {}
        self.pd_output = padl.transforms._OutputSlicer(self)
        self.set_output_slice()

    def replace_outnode(self, old_node, new_node):
        if old_node in self.out_node:
            self.out_node = [node if node != old_node else new_node for node in self.out_node]
            self._output_slice = {(new_node if key == old_node else key): val for key, val in self._output_slice.items()}

            new_node.connect_innode(self)
            old_node.remove_innode(self)

    def replace_innode(self, old_node, new_node):
        if old_node in self.in_node:
            self.in_node = [node if node != old_node else new_node for node in self.in_node]
            if self._input_args is not None:
                self._input_args = {(new_node if key == old_node else key): val for key, val in self._input_args.items()}

            new_node.connect_outnode(self)
            old_node.remove_outnode(self)

    def remove_outnode(self, node):
        if node in self.out_node:
            self.out_node = [n_ for n_ in self.out_node if n_ != node]
            node.remove_innode(self)

    def remove_innode(self, node):
        if node in self.in_node:
            self.in_node = [n_ for n_ in self.in_node if n_ != node]
            node.remove_outnode(self)

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
            self._output_slice[node] = self.get_output_slice()

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
        slice = self._output_slice[out_node]
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
            self._create_input_args_dict()
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
        self.transform_nodes = []
        self.node_dict = dict()

    def _store_transform_nodes(self, transforms):
        self.nodes.append(self.input_node)
        for t_ in transforms:
            if isinstance(t_, Graph):
                graph_copy = deepcopy(t_)
                self.nodes.extend(graph_copy.nodes)
                self.nodes.append(graph_copy)
            elif isinstance(t_, Node):
                node_copy = deepcopy(t_)
                self.nodes.append(node_copy)
            else:
                self.nodes.append(Node(t_))
            self.transform_nodes.append(self.nodes[-1])
        self.nodes.append(self.output_node)
        self.node_dict = {n_.name_id: n_ for n_ in self.nodes}

    def connect_innode(self, node):
        if node not in self.input_node.in_node:
            self.input_node.in_node.append(node)
            node.connect_outnode(self.input_node)
            self.input_node._create_input_args_dict()

    def connect_outnode(self, node):
        if node not in self.output_node.out_node:
            self.output_node.connect_outnode(node)
            node.connect_innode(self.output_node)
            self.output_node._output_slice[node] = self.get_output_slice()

    def compose(self, transforms):
        self._store_transform_nodes(transforms)

        node = self.transform_nodes[0]
        node.connect_innode(self.input_node)

        for next_node in self.transform_nodes[1:]:
            next_node.connect_innode(node)
            node = next_node
        next_node.connect_outnode(self.output_node)

    def rollout(self, transforms):
        self._store_transform_nodes(transforms)

        for node in self.transform_nodes:
            node.connect_innode(self.input_node)
            node.connect_outnode(self.output_node)

    def parallel(self, transforms):
        self._store_transform_nodes(transforms)

        for indx, node in enumerate(self.transform_nodes):
            node.connect_innode(self.input_node.pd_output[indx])
            node.connect_outnode(self.output_node)

    def pd_call_node(self, args, in_node=None):
        output = self.input_node.pd_call_node(args, in_node)

        return output

    def __hash__(self):
        return hash(repr(self))

    def __deepcopy__(self, memo):
        """Deepcopy of Nodes"""
        def _copy_nodes(_copy_graph, inp_node, copy_inp_node, copied_node_dict={}):
            for node in inp_node.out_node:
                if node in copied_node_dict:
                    copy_node = copied_node_dict[node]
                else:
                    copy_node = deepcopy(node)
                    copy_node.in_node = []
                    copy_node.out_node = []
                    copied_node_dict[node] = copy_node

                if isinstance(copy_node, Graph):
                    _copy_graph.nodes.extend(copy_node.nodes)
                _copy_graph.nodes.append(copy_node)

                copy_inp_node.connect_outnode(copy_node)
                copy_inp_node._output_slice[copy_node] = inp_node._output_slice.get(node, None)
                _copy_nodes(_copy_graph, node, copy_node, copied_node_dict)

            if len(inp_node.out_node) == 0:
                _copy_graph.output_node = copy_inp_node

        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                self.transform)
            _copy.in_node = copy(self.in_node)
            _copy.out_node = copy(self.out_node)

            _copy.nodes.append(_copy.input_node)
            copied_node_dict = {self.input_node: _copy.input_node,
                                self.output_node: _copy.output_node}
            _copy_nodes(_copy, self.input_node, _copy.input_node, copied_node_dict)

            _copy.node_dict = {n_.name_id: n_ for n_ in _copy.nodes}
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

    def list_all_paths(self, start_node_name=None, end_node_name=None):
        if start_node_name is None:
            start_node_name = self.input_node.name_id
        if end_node_name is None:
            end_node_name = self.output_node.name_id
        return nx.all_simple_paths(self.networkx_graph, start_node_name, end_node_name)

    def count_batchify_unbatchify(self):
        for path in self.list_all_paths():
            batch_counter = 0
            unbatch_counter = 0
            for node_name in path:
                node = self[node_name]
                if isinstance(node.transform, padl.Batchify):
                    batch_counter += 1
                    assert batch_counter < 2, f"Error: Path contains more than 1 batchify : {path}"
                elif isinstance(node.transform, padl.Unbatchify):
                    unbatch_counter += 1
                    assert unbatch_counter < 2, f"Error: Path contains more than 1 unbatchify : {path}"
        return

    def __getitem__(self, item):
        if isinstance(item, (slice, int)):
            return self.nodes[item]
        if isinstance(item, str):
            return self.node_dict[item]
        raise f'Error: invalid index {item}, accept int, slice or string'

    def add_edge(self, *connection_tuple):
        """ Currently this is really badly implemented that
        first creates connection in self and then copies self
        and removes connection

        Once we have good naming system, so name are kept when deepcopying,
        this can be made simpler by copying graph and using the same name
        to create edge in copy graph"""
        for in_node_name, out_node_name in connection_tuple:
            in_node = self[in_node_name]
            out_node = self[out_node_name]
            in_node.connect_outnode(out_node)
        copy_graph = deepcopy(self)
        for in_node_name, out_node_name in connection_tuple:
            in_node = self[in_node_name]
            out_node = self[out_node_name]
            in_node.out_node.remove(out_node)
            out_node.in_node.remove(in_node)
        return copy_graph
