
import networkx as nx
import padl


class Graph:

    def __init__(self):
        self._input_connected = False
        self._node_id = 0
        self.nodes = []

        self.input_node = _InputNode()
        self.input_node.update_graph(self)

        #self.output_node = _OutputNode()
        #self.output_node.update_graph(self)

        self.in_nodes = []
        self.out_nodes = []

        self.networkx_graph = None

    def __call__(self, args):
        if isinstance(args, Node):
            self.add_node(args, connect_input=True)
        if isinstance(args, (list, tuple)):
            for node_ in args:
                self(node_)
        return

    def generate_node_id(self):
        self._node_id += 1
        return self._node_id-1

    def connect_to_input_node(self, node):
        self.input_node.insert_output_node(node)
        self._input_connected = True

    def add_node(self, node, connect_input=False):
        if node in self.nodes:
            return

        self.nodes.append(node)
        node.update_graph(self)
        node.id = self.generate_node_id()

        if connect_input or \
                (
                not isinstance(node, (_InputNode, _OutputNode))
                and not self._input_connected
                ):
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
        inbuilt_kwargs = dict(
            with_labels=with_labels,
            node_size=3500,
            node_shape="s",
            width=1,
            alpha=0.9,
            linewidths=3,
            node_color='lightblue'
        )
        inbuilt_kwargs.update(kwargs)
        return nx.draw(self.networkx_graph, with_labels=with_labels, **inbuilt_kwargs)


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
        if self in node.out_nodes:
            return
        node.insert_output_node(self)

    def _insert_graph_at_end(self, graph):
        self.unflattened_out_nodes.append(graph)
        inputnode = graph.input_node
        for node in inputnode.out_nodes_iterator():
            self.insert_output_node(node)

    def insert_output_node(self, node):
        if isinstance(node, Graph):
            self._insert_graph_at_end(node)
            return
        self.update_graph(node.graph)
        if node not in self.out_nodes:
            self.out_nodes.append(node)
        if self in node.out_nodes:
            return
        node.insert_input_node(self)

    def in_nodes_iterator(self):
        for node in self.in_nodes:
            yield node

    def out_nodes_iterator(self):
        for node in self.out_nodes:
            yield node

    def __call__(self, args):
        if isinstance(args, Node):
            assert args.graph
            self.insert_input_node(args)
        if isinstance(args, (list, tuple)):
            for node_ in args:
                self(node_)
        return

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


class _OutputNode(Node):
    def __init__(self):
        super().__init__(padl.Identity(), 'OutputNode')

    def update_graph(self, graph):
        if graph is None:
            return
        if self.graph == graph:
            return
        self.graph = graph
