import networkx as nx
import padl
from IPython.display import Image


class Node:
    _id = 0

    def __init__(self, transform=None, name=None):
        self.id = self.generate_id()
        self.transform = transform
        self.name = name
        if self.name is None and self.transform is not None:
            self.name = self.transform.pd_name
        if self.name is not None:
            self.name += ' '+str(self.id)
        self.in_node = []
        self.out_node = []
        self.out_index = None
        self._input_args = None

    @classmethod
    def generate_id(cls):
        cls._id += 1
        return cls._id

    def __call__(self, args):
        # self._output = self.transform(self.args)
        # if self.out_index is not None:
        #     return self.output[self.out_index]
        return self.pd_call_node(args)

    def __hash__(self):
        return hash(repr(self))

    def connect_innode(self, node):
        if node not in self.in_node:
            self.in_node.append(node)
            node.connect_outnode(self)

    def connect_outnode(self, node):
        if node not in self.out_node:
            self.out_node.append(node)
            node.connect_innode(self)

    def _create_input_args_dict(self):
        """Update dictionary of args for input"""
        self._input_args = {node: {'args': None, 'updated': False} for node in self.in_node}
        if len(self._input_args) == 0:
            self._input_args[None] = {'args': None, 'updated': False}

    def _register_input_args(self, args, in_node):
        """Register the input args"""
        assert in_node in self._input_args.keys(), f"{in_node} not connected to {self}"
        self._input_args[in_node]['args'] = args
        self._input_args[in_node]['updated'] = True

        if all([in_args['updated'] for _, in_args in self._input_args.items()]):
            return True
        return False

    def pd_call_transform(self):
        args = [in_args['args'] for _, in_args in self._input_args.items()]
        if len(args) == 1:
            return self.transform.pd_call_transform(args[0])
        return self.transform.pd_call_transform(tuple(args))

    def pd_call_node(self, args, in_node=None):
        if self._input_args is None:
            self._create_input_args_dict()
        if self._register_input_args(args, in_node):
            output = self.pd_call_transform()
            for out_node in self.out_node:
                output = out_node.pd_call_node(output, self)
            return output


class Graph(Node):
    def __init__(self, transforms=None, connection=None):
        super().__init__()
        self.transforms = transforms
        self.input = Node(padl.Identity(), name='Input')
        self.output = Node(padl.Identity(), name='Output')
        self._input_args = None
        self.networkx_graph = None
        self.nodes = []

    def _store_transform_nodes(self, transforms):
        for t_ in transforms:
            if isinstance(t_, Node):
                self.nodes.append(t_)
            else:
                self.nodes.append(Node(t_))

    def connect_innode(self, node):
        if node not in self.input.in_node:
            self.input.in_node.append(node)
            node.connect_outnode(self.input)

    def connect_outnode(self, node):
        if node not in self.out_node:
            self.output.out_node.append(node)
            node.connect_innode(self.output)

    def compose(self, transforms):
        self._store_transform_nodes(transforms)

        node = self.nodes[0]
        node.connect_innode(self.input)

        for next_node in self.nodes[1:]:
            next_node.connect_innode(node)
            node = next_node

        next_node.connect_outnode(self.output)

    def rollout(self, transforms):
        self._store_transform_nodes(transforms)

        for node in self.nodes:
            node.connect_innode(self.input)
            node.connect_outnode(self.output)

    def pd_call_node(self, args, in_node=None):
        output = self.input.pd_call_node(args, in_node)

        return output

    def __hash__(self):
        return hash(repr(self))
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
            networkx_graph.add_node(node.name, node=node)
            networkx_graph.add_edge(innode.name, node.name)
            self._add_to_networkx_graph_name(node, networkx_graph)

    def convert_to_networkx(self, with_name=False):
        node = self.input
        self.networkx_graph = nx.DiGraph()

        if with_name:
            self.networkx_graph.add_node(node.name, node=node)
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
