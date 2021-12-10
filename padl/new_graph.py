
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
        if node in self.nodes:
            return

        if isinstance(node, Graph):
            self._add_graph_as_node(node)
            return

        self.nodes.append(node)
        node.id = self.generate_node_id()

        node.update_graph(self)

        if connect_output or len(node.out_nodes) == 0:
            self.connect_to_output_node(node)

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

    def draw(self, with_labels=True, **kwargs):
        if self.networkx_graph is None:
            self.convert_to_networkx()
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

    def __init__(self, transform=None, name=None, graph=None):
        self.id = None
        self._graph = None
        self.in_nodes = []
        self.out_nodes = []

        self.unflattened_in_nodes = []
        self.unflattened_out_nodes = []

        self.name = name
        self.transform = transform
        self.update_graph(graph)
        self._update_input_args()

    def _update_input_args(self):
        """Update dictionary of args for input"""
        self._input_args = {node: {'args': None, 'updated': False} for node in self.in_nodes}

    def _register_input_args(self, args, in_node):
        """Register the input args"""
        assert in_node in self.in_nodes, f"{in_node} not connected to {self}"
        self._input_args[in_node]['args'] = args
        self._input_args[in_node]['updated'] = True

        if all([in_args['updated'] for _, in_args in self._input_args.items()]):
            return True
        return False

    def __call__(self, args, *, in_node=None):

        if isinstance(args, Node):
            assert args.graph
            self.insert_input_node(args)
            return

        if isinstance(args, (list, tuple)) and\
            any(map(lambda x: isinstance(x, Node), args)):
            for node in args:
                self(node)
            return

        if self._register_input_args(args, in_node):
            output = self.transform.pd_call_transform(args)
            for out_node in self.out_nodes_iterator():
                out_node(output, in_node=self)
        return

    def reset_input_node(self):
        self.in_nodes = []

    def reset_output_node(self):
        self.out_nodes = []

    def reset_graph(self):
        self.graph = None

    def insert_input_node(self, node):
        self.update_graph(node.graph)
        if node not in self.in_nodes:
            self.in_nodes.append(node)
        node.insert_output_node(self)
        if self not in node.out_nodes:
            node.insert_output_node(self)
        return node

    def _insert_graph_at_end(self, graph):
        self.unflattened_out_nodes.append(graph)
        self.insert_output_node(graph.input_node)
        graph.move_node_to_new_graph(self.graph)
        return graph.output_node

    def insert_output_node(self, node):
        """Insert nodes/list or tuple of nodes to out"""
        if isinstance(node, Graph):
            return self._insert_graph_at_end(node)
        # if isinstance(node, _OutputNode):

        self.update_graph(node.graph)
        if node not in self.out_nodes:
            self.out_nodes.append(node)
        if self not in node.in_nodes:
            node.insert_input_node(self)
        return node

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
