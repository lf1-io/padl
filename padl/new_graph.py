import padl


class Graph:

    def __init__(self):
        self._node_id = 0
        self.nodes = []

        self.input_node = _InputNode()
        self.output_node = _OutputNode()
        self.input_node.update_graph(self)
        self.output_node.update_graph(self)

        self.in_nodes = []
        self.out_nodes = []

    def generate_node_id(self):
        self._node_id += 1
        return self._node_id-1

    def add_node(self, node, connect_input=False):
        if node in self.nodes:
            return

        self.nodes.append(node)
        node.update_graph(self)
        node.id = self.generate_node_id()
        if connect_input:
            self.input_node.insert_output_node(node)


class Node:

    def __init__(self, transform=None, name=None):
        self.id = None
        self._graph = None
        self.in_nodes = []
        self.out_nodes = []
        self.name = name
        self.transform = transform

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

    def insert_output_node(self, node):
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
            self.insert_input_node(args)

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
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

    def __call__(self, args):
        if isinstance(args, Node):
            self.in_nodes.append(args)
            args.out_nodes.append(self)
            self.update_graph(args.graph)
        return


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

    def update_graph(self,graph):
        if graph is None:
            return
        if self.graph == graph:
            return
        self.graph = graph
