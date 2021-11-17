

class Graph:
    _node_id = 0

    def __init__(self, transform=None):
        self.nodes = []
        self.start_node = None
        if transform is not None:
            self.start_node = Node(transform, graph=self, id=self._node_id)
        self.input_node = InputNode(graph=self)
        self.output_node = OutputNode(graph=self)
        self.nodes.extend([self.input_node, self.output_node])

        if self.start_node:
            self.start_node.input_node = [self.input_node]
            self.start_node.output_node = [self.output_node]

    def add_node(self, node):
        if node in self.nodes:
            return
        self.nodes.append(node)
        node.graph = self
        node.update_id(self.get_next_node_id())

    @classmethod
    def get_next_node_id(cls):
        cls._node_id += 1
        return cls._node_id


class Node:

    def __init__(self, transform, graph=None, id=None, input=None, output=None):
        self._graph = None
        if graph is not None:
            self.graph = graph

        self.id = id if id is not None else 0
        if self.graph is None:
            self.input_node = self._init_input(input)
            self.output_node = self._init_output(output)
        self._input_args = {node: None for node in self.input_node}
        self._updated_args = []
        self.transform = transform

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, new_graph):
        self._graph = new_graph
        if new_graph is None:
            return
        new_graph.add_node(self)

    def update_id(self, new_id):
        self.id = new_id

    def outnode_iter(self):
        for node in self.output_node:
            yield node

    def innode_iter(self):
        for node in self.input_node:
            yield node

    def _drop_inbuilt_output_node(self):
        out_nodes = [out_node for out_node in self.output_node if not isinstance(out_node, OutputNode)]
        self.output_node = out_nodes

    def _drop_inbuilt_input_node(self):
        in_nodes = [in_node for in_node in self.input_node if not isinstance(in_node, InputNode)]
        self.input_node = in_nodes

    def update_output_node(self, node):
        self._drop_inbuilt_output_node()
        if self.graph != node.graph:
            self.graph.add_node(node)
        self.output_node = [node]
        if self in node.input_node:
            return
        node.insert_input_node(self)

    def insert_output_node(self, node):
        self._drop_inbuilt_output_node()
        if self.graph != node.graph:
            self.graph.add_node(node)
        self.output_node.append(node)
        if self in node.input_node:
            return
        node.insert_input_node(self)

    def update_input_node(self, node):
        self._drop_inbuilt_input_node()
        if self.graph != node.graph:
            self.graph.add_node(node)
        self.input_node = [node]
        if self in node.output_node:
            return
        node.insert_output_node(self)

    def insert_input_node(self, node):
        self._drop_inbuilt_input_node()
        if self.graph != node.graph:
            self.graph.add_node(node)
        self.input_node.append(node)
        if self in node.output_node:
            return
        node.insert_output_node(self)

    def _init_input(self, input):
        if input is None:
            return [InputNode(self.graph)]
        return [input]

    def _init_output(self, output):
        if output is None:
            return [OutputNode(self.graph)]
        return [output]

    def __hash__(self):
        return hash(repr(self))

    def _check_input_complete(self):
        return all([True if in_node in self._updated_args else False for in_node in self.input_node])

    def _register_input_args(self, args, node):
        self._input_args[node] = args
        self._updated_args.append(node)

    def _pd_call_transform(self):
        args = list(self._input_args.values())
        if len(args) == 1:
            return self.transform._pd_call_transform(args[0])
        return self.transform._pd_call_transform(tuple(args))

    def __call__(self, args, node):
        self._register_input_args(args, node)
        if self._check_input_complete():
            return self._pd_call_transform()
        return



class InputNode(Node):

    def __init__(self, graph=None, id=None):
        self.graph = graph
        self.id = id if id is not None else 0


class OutputNode(Node):

    def __init__(self, graph=None, id=None):
        self.graph = graph
        self.id = id if id is not None else 0

