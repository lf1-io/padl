import padl


class Graph:
    _node_id = 0

    def __init__(self):
        self.nodes = []
        self.input_node = InputNode(graph=self)
        self.output_node = OutputNode(graph=self, in_node=self.input_node)
        self.input_node.out_nodes = [self.output_node]

    def add_node(self, node):
        if node in self.nodes:
            return
        self.nodes.append(node)
        node.update_graph(self)
        node.update_id(self.get_next_node_id())

    def get_next_node_id(self):
        self._node_id += 1
        return self._node_id


class BaseNode:

    def __init__(self, transform, graph=None, id=None, in_node=None, out_node=None):
        self._graph = None
        if graph is not None:
            self.graph = graph
            id = id if id is not None else self.graph.get_next_node_id()

        self.transform = transform
        self.id = id

        self.in_nodes = [in_node]
        self.out_nodes = [out_node]

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, new_graph):
        if new_graph is None:
            return
        if self.graph == new_graph:
            return
        self._graph = new_graph
        new_graph.add_node(self)

    def update_graph(self, new_graph):
        if self.graph == new_graph:
            return
        self.graph = new_graph
        if self.graph is None:
            return

    def update_id(self, new_id):
        self.id = new_id

    def outnode_iter(self):
        for node in self.out_nodes:
            yield node

    def innode_iter(self):
        for node in self.in_nodes:
            yield node


class Node(BaseNode):

    def __init__(self, transform, graph=None, id=None, in_node=None, out_node=None):
        super().__init__(transform, graph, id, in_node, out_node)
        if self.graph is None:
            self.in_node = self._init_input(in_node)
            self.out_node = self._init_output(out_node)
        self._input_args = {node: None for node in self.in_node}
        self._updated_args = []
        self.transform = transform

    def _drop_inbuilt_output_node(self):
        out_nodes = [out_node for out_node in self.out_node if not isinstance(out_node, OutputNode)]
        self.out_node = out_nodes

    def _drop_inbuilt_input_node(self):
        in_nodes = [in_node for in_node in self.in_node if not isinstance(in_node, InputNode)]
        self.in_node = in_nodes

    def update_output_node(self, node):
        self._drop_inbuilt_output_node()
        if self.graph != node.graph:
            self.graph.add_node(node)
        self.out_node = [node]
        if self in node.in_node:
            return
        node.insert_input_node(self)

    def insert_output_node(self, node):
        self._drop_inbuilt_output_node()
        if self.graph != node.graph:
            self.graph.add_node(node)
        self.out_node.append(node)
        if self in node.in_node:
            return
        node.insert_input_node(self)

    def update_input_node(self, node):
        self._drop_inbuilt_input_node()
        if self.graph != node.graph:
            self.graph.add_node(node)
        self.in_node = [node]
        if self in node.out_node:
            return
        node.insert_output_node(self)

    def insert_input_node(self, node):
        self._drop_inbuilt_input_node()
        if self.graph != node.graph:
            self.graph.add_node(node)
        self.in_node.append(node)
        if self in node.out_node:
            return
        node.insert_output_node(self)

    def _init_input(self, input):
        if input is None:
            return [InputNode(self.graph)]
        input.insert_output_node(self)
        return [input]

    def _init_output(self, output):
        if output is None:
            return [OutputNode(self.graph)]
        output.insert_input_node(self)
        return [output]

    def __hash__(self):
        return hash(repr(self))

    def _check_input_complete(self):
        return all([True if in_node in self._updated_args else False for in_node in self.in_node])

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



class InputNode(BaseNode):

    def __init__(self, graph=None, id=None, out_node=None):
        super().__init__(padl.Identity(), graph=graph, id=id, out_node=out_node)


class OutputNode(BaseNode):

    def __init__(self, graph=None, id=None, in_node=None):
        super().__init__(padl.Identity(), graph=graph, id=id, in_node=in_node)
