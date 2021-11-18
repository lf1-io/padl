
class Graph:

    def __init__(self):
        self._node_id = 0
        self.nodes = []
        self.input_node = []

    def generate_node_id(self):
        self._node_id += 1
        return self._node_id-1

    def add_node(self, node):
        if node in self.nodes:
            return
        self.nodes.append(node)
        node.update_graph(self)
        node.id = self.generate_node_id()


class Node:

    def __init__(self, transform=None, name=None):
        self.id = None
        self._graph = None
        self.in_nodes = []
        self.out_nodes = []
        self.name = None
        self.transform = transform

    @property
    def graph(self):
        return self._graph

    def update_name(self, new_name):
        self.name = new_name

    def update_graph(self, graph):
        if graph is None:
            return
        if self._graph == graph:
            return
        self._graph = graph
        graph.add_node(self)

    def __hash__(self):
        return hash(repr(self))

    def __call__(self, args):
        if isinstance(args, Node):
            self.in_nodes.append(args)
            args.out_nodes.append(self)
            self.update_graph(args.graph)
        return
