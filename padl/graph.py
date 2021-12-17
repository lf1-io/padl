import padl


class Node:
    def __init__(self, transform):
        self.transform = transform
        self.in_node = []
        self.out_node = []
        self.out_index = None
        self._input_args = None

    def __call__(self, args):
        self._output = self.transform(self.args)
        if self.out_index is not None:
            return self.output[self.out_index]
        return self._output

    def __hash__(self):
        return hash(repr(self))

    def _update_input_args(self):
        """Update dictionary of args for input"""
        self._input_args = {node: {'args': None, 'updated': False} for node in self.in_node}
        if len(self._input_args) == 0:
            self._input_args[None] = {'args': None, 'updated': False}

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
        output = self.transform(args)

        for out_node in self.out_node:
            out_node.pd_call_node(output, in_node)


class Graph(Node):
    def __init__(self, transforms=None, connection=None):
        self.transforms = transforms
        self.input = Node(padl.Identity())
        self.output = Node(padl.Identity())

    def _store_transform_nodes(self, transforms):
        self.nodes = [Node(t_) for t_ in transforms]

    def compose(self, transforms):
        self._store_transform_nodes(transforms)

        node = self.nodes[0]
        node.in_node.append(self.input)
        self.input.out_node.append(node)

        for next_node in self.nodes[1:]:
            node.out_node.append(next_node)
            next_node.in_node.append(node)

    def rollout(self, transforms):
        self._store_transform_nodes(transforms)

        for node in self.nodes:
            node.in_node.append(self.input)
            node.out_node.append(self.output)

    def __hash__(self):
        return hash(repr(self))
    """
    def parallel(self, transforms):
        self._store_transform_nodes(transforms)
        
        for ind, node in enumerate(self.nodes):
    """
