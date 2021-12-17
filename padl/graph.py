import padl


class Node:
    def __init__(self, transform):
        self.transform = transform
        self.in_node = []
        self.out_node = []
        self.out_index = None

    def __call__(self, args):
        self._output = self.transform(self.args)
        if self.out_index is not None:
            return self.output[self.out_index]
        return self._output


class Graph:
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

    """
    def parallel(self, transforms):
        self._store_transform_nodes(transforms)
        
        for ind, node in enumerate(self.nodes):
    """
