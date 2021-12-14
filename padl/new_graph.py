
from copy import copy, deepcopy
import networkx as nx
import padl


class Graph:

    def __init__(self):
        self._input_connected = False
        self.id = None
        self._node_id = 0
        self.nodes = []

        self.input_node = _InputNode()
        self.output_node = _OutputNode()

        self.input_node.update_graph(self)
        self.output_node.update_graph(self)

        self.in_nodes = []
        self.out_nodes = []

        self.networkx_graph = None

    def build_output_connections(self):
        """Connect all dangling ends to output"""
        for node_ in self.nodes:
            if len(node_.out_nodes) == 0 and node_ != self.output_node:
                self.connect_to_output_node(node_)

    def move_node_to_new_graph(self, new_graph):
        """Move all nodes of this graph to New graph

        Except Input and Output node
        """
        for node_ in self.nodes:
            # if isinstance(node_, (_InputNode, _OutputNode)):
            #    continue
            new_graph.add_node(node_)

    def update_graph(self, new_graph):
        """Move all nodes to new graph"""
        self.move_node_to_new_graph(new_graph)

    def __call__(self, args):
        """Call the graph - its node from input till output"""

        if isinstance(args, Node):
            self.add_node(args, connect_input=True)
            return

        if isinstance(args, (list, tuple)) and\
            any(map(lambda x: isinstance(x, Node), args)):
            for node in args:
                self(node)
            return

        self._update_input_args()
        self.input_node(args)
        return self._pack_output()

    def _pack_output(self):
        """Pack the final output of graph"""
        output = []
        for out in self.output_node.input_args:
            output.append(out['args'])
        return tuple(output)

    def _update_input_args(self):
        for node in self.nodes:
            node._update_input_args()

    def generate_node_id(self):
        self._node_id += 1
        return self._node_id-1

    def connect_to_input_node(self, node):
        self.input_node.insert_output_node(node)
        self._input_connected = True

    def connect_to_output_node(self, node):
        if node == self.output_node:
            return
        self.output_node.insert_input_node(node)

    def _add_graph_as_node(self, graph):
        self.nodes.append(graph)
        graph.id = self.generate_node_id()
        graph.update_graph(self)

    def add_node(self,
                 node,
                 connect_input=False,
                 connect_output=False):
        return
        if node in self.nodes:
            return

        if isinstance(node, Graph):
            node.move_node_to_new_graph(self)
            return

        self.nodes.append(node)
        node.id = self.generate_node_id()

        node.update_graph(self)

        if connect_input:
            # or \
            # (
            # not isinstance(node, (_InputNode, _OutputNode))
            # and not self._input_connected
            # ):
            self.input_node.insert_output_node(node)
            self._input_connected = True

    def _add_to_networkx_graph_id(self, innode):
        for outnode in innode.out_nodes_iterator():
            self.networkx_graph.add_node(outnode.id, node=outnode)
            self.networkx_graph.add_edge(innode.id, outnode.id)
            self._add_to_networkx_graph_id(outnode)

    def _add_to_networkx_graph_name(self, innode):
        for outnode in innode.out_nodes_iterator():
            self.networkx_graph.add_node(outnode.name, node=outnode)
            self.networkx_graph.add_edge(innode.name, outnode.name)
            self._add_to_networkx_graph_name(outnode)

    def convert_to_networkx(self, with_name=False):
        self.networkx_graph = nx.DiGraph()
        if with_name:
            self.networkx_graph.add_node(self.input_node.name, node=self.input_node)
            self._add_to_networkx_graph_name(self.input_node)

            return

        self.networkx_graph.add_node(self.input_node.id, node=self.input_node)
        self._add_to_networkx_graph_id(self.input_node)
        return

    def draw(self, with_name=False, with_labels=True, **kwargs):
        if self.networkx_graph is None:
            self.convert_to_networkx(with_name=with_name)
        inbuilt_kwargs = dict(
            with_labels=with_labels,
            node_size=3500,
            node_shape="s",
            width=1,
            alpha=0.9,
            linewidths=3,
            node_color='lightblue',
        )
        inbuilt_kwargs.update(kwargs)
        return nx.draw(self.networkx_graph, **inbuilt_kwargs)


class Node:
    _id = 0

    def __init__(self, transform=None, name=None, graph=None, id=None):
        self.id = id if id is not None else self.generate_id()
        self._graph = None
        self.in_nodes = []
        self.out_nodes = []

        self.unflattened_in_nodes = []
        self.unflattened_out_nodes = []

        self.name = name
        self.transform = transform
        self.update_graph(graph)
        self._update_input_args()
        self.output = []
        self.networkx_graph = None

    @classmethod
    def generate_id(cls):
        cls._id += 1
        return cls._id

    def _update_input_args(self):
        """Update dictionary of args for input"""
        self._input_args = {node: {'args': None, 'updated': False} for node in self.in_nodes}
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

    def _assign_graph(self, node):
        if self.graph:
            node.update_graph(self.graph)
            return self.graph
        if node.graph:
            self.update_graph(node.graph)
            return node.graph
        graph = Graph()
        self.update_graph(graph)
        node.update_graph(graph)
        return

    def __deepcopy__(self, memo):
        """Deepcopy of Nodes"""
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                self.transform,
                self.name,
                self._graph,
                self.id)
            _copy.in_nodes = copy(self.in_nodes)
            _copy.out_nodes = copy(self.out_nodes)
            memo[id_self] = _copy
        return _copy

    def pd_call_transform(self):
        args = [in_args['args'] for _, in_args in self._input_args.items()]
        if len(args) == 1:
            return self.transform.pd_call_transform(args[0])
        return self.transform.pd_call_transform(tuple(args))

    def pd_call_node(self, args, in_node):
        """Call this node"""
        if self._register_input_args(args, in_node):
            transform_output = self.pd_call_transform()
            if self.name == 'OutputNode':
                self.output.append(transform_output)
            for out_node in self.out_nodes_iterator():
                # import pdb; pdb.set_trace()
                out_node.pd_call_node(transform_output, in_node=self)
                # import pdb;pdb.set_trace()
                # print('done')

    def pd_call_node_graph(self, args, in_node):
        input_node = self.get_input_node()
        input_node.pd_call_node(args, in_node)
        return tuple(self.output)

    def __call__(self, args, *, in_node=None):

        if isinstance(args, Node):
            self._assign_graph(args)
            if args.name == 'OutputNode':
                self.insert_output_node(args)
                return
            self.insert_input_node(args)
            return

        if isinstance(args, (list, tuple)) and\
            any(map(lambda x: isinstance(x, Node), args)):
            for node in args:
                self(node)
            return

        self.pd_call_node_graph(args, in_node)
        return tuple([v['args'] for _, v in self._input_args.items()])
        """
        if self._register_input_args(args, in_node):
            transform_output = self.pd_call_transform()
            output = []
            for out_node in self.out_nodes_iterator():
                # import pdb; pdb.set_trace()
                output.append(out_node(transform_output, in_node=self))
                # import pdb;pdb.set_trace()
                # print('done')
        """       # return output

    def reset_input_node(self):
        self.in_nodes = []

    def reset_output_node(self):
        self.out_nodes = []

    def reset_graph(self):
        self.graph = None

    def insert_input_node(self, node):
        self.update_graph(node.graph)
        # import pdb;pdb.set_trace()
        if node not in self.in_nodes:
            # import pdb; pdb.set_trace()
            self.in_nodes.append(node)
        if self not in node.out_nodes:
            # import pdb; pdb.set_trace()
            node.insert_output_node(self)
        self._update_input_args()
        return node

    def insert_output_node(self, node):
        """Insert nodes/list or tuple of nodes to out"""
        if isinstance(node, Graph):
            return self._insert_graph_at_end(node)
        # if isinstance(node, _OutputNode):

        self.update_graph(node.graph)
        if node not in self.out_nodes:
            # import pdb; pdb.set_trace()
            self.out_nodes.append(node)
        # import pdb;pdb.set_trace()
        if self not in node.in_nodes:
            # import pdb; pdb.set_trace()
            node.insert_input_node(self)
        return node

    def _insert_graph_at_end(self, graph):
        self.unflattened_out_nodes.append(graph)
        self.insert_output_node(graph.input_node)
        graph.move_node_to_new_graph(self.graph)
        return graph.output_node


    def in_nodes_iterator(self):
        for node in self.in_nodes:
            yield node

    def out_nodes_iterator(self):
        for node in self.out_nodes:
            yield node

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        if self not in graph.nodes:
            graph.add_node(self)
        self._graph = graph

    def update_name(self, new_name):
        self.name = new_name

    def update_graph(self, graph):
        if graph is None:
            return
        if self.graph == graph:
            return
        self.graph = graph
        graph.add_node(self)

    def __hash__(self):
        return hash(repr(self))

    def _add_to_networkx_graph_id(self, outnode, networkx_graph):
        for innode in outnode.in_nodes_iterator():
            networkx_graph.add_node(innode.id, node=innode)
            networkx_graph.add_edge(innode.id, outnode.id)
            self._add_to_networkx_graph_id(innode, networkx_graph)

    def _add_to_networkx_graph_name(self, outnode, networkx_graph):
        for innode in outnode.in_nodes_iterator():
            networkx_graph.add_node(innode.name, node=innode)
            networkx_graph.add_edge(innode.name, outnode.name)
            self._add_to_networkx_graph_name(innode, networkx_graph)

    def convert_to_networkx(self, with_name=False):
        node = self
        self.networkx_graph = nx.DiGraph()
        if with_name:
            self.networkx_graph.add_node(node.name, node=node)
            self._add_to_networkx_graph_name(node, self.networkx_graph)
            return self.networkx_graph

        self.networkx_graph.add_node(node.id, node=node)
        self._add_to_networkx_graph_id(node, self.networkx_graph)
        return self.networkx_graph

    def draw(self, with_name=False, with_labels=True, **kwargs):
        self.convert_to_networkx(with_name=with_name)
        inbuilt_kwargs = dict(
            with_labels=with_labels,
            node_size=3500,
            node_shape="s",
            width=1,
            alpha=0.9,
            linewidths=3,
            node_color='lightblue',
        )
        inbuilt_kwargs.update(kwargs)
        return nx.draw(self.networkx_graph, **inbuilt_kwargs)

    def get_input_node(self):
        def _get_input_node(node):
            out = None
            for innode in node.in_nodes:
                if len(innode.in_nodes) == 0:
                    return innode
                out = _get_input_node(innode)
            return out

        return _get_input_node(self)


class _InputNode(Node):
    def __init__(self):
        super().__init__(padl.Identity(), 'InputNode')

    def update_graph(self, graph):
        if graph is None:
            return
        if self.graph == graph:
            return
        self.graph = graph

    def _register_input_args(self, args, in_node=None):
        return True


class _OutputNode(Node):
    def __init__(self):
        super().__init__(padl.Identity(), 'OutputNode')

    def update_graph(self, graph):
        if graph is None:
            return
        if self.graph == graph:
            return
        self.graph = graph


