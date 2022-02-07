from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from typing import Any

# TODO: remove
from IPython.display import Image
import networkx as nx



@dataclass
class _Edge:
    out: Any
    in_: Any

    def __str__(self):
        def fmt(x):
            if x is None:
                return ':'
            return x
        return f'{fmt(self.out)} - {fmt(self.in_)}'

    def __lt__(self, other):
        return self.out is None or self.out < other.out


class _Node:
    _id = 0

    def __init__(self):
        self.id = self._next_id()

    def __repr__(self):
        return str(self.id)

    def copy(self):
        new = copy(self)
        new.id = self._next_id()
        return new

    def _next_id(cls):
        res = _Node._id
        _Node._id += 1
        return res


class _Graph:
    def __init__(self, edges=None, input_node=None, output_node=None):
        if edges is None:
            edges = {}
        self.edges = defaultdict(dict)
        self.edges.update(edges)
        self._parents = None
        self.update_parents()
        if input_node is None:
            self.input_node = _Node()
        else:
            self.input_node = input_node
        if output_node is None:
            self.output_node = _Node()
        else:
            self.output_node = output_node

    def copy(self, memo=None):
        if memo is None:
            memo = {}
        new_edges = defaultdict(dict)
        for node, children in self.edges.items():
            if node not in memo:
                memo[node] = node.copy()
            for child, edge in children.items():
                if child not in memo:
                    memo[child] = child.copy()
                new_edges[memo[node]][memo[child]] = edge
        return _Graph(new_edges,
                      memo.get(self.input_node, self.input_node.copy()),
                      memo.get(self.output_node, self.output_node.copy()))

    def update_parents(self):
        self._parents = defaultdict(set)
        for node, children in self.edges.items():
            for child in children:
                self._parents[child].add(node)

    def children(self, edge):
        return sorted(self.edges[edge], key=self.edges[edge].get)

    def parents(self, edge):
        return self._parents[edge]

    @property
    def in_nodes(self):
        return self.children(self.input_node)

    @property
    def out_nodes(self):
        return self.parents(self.output_node)

    def in_edges(self, node):
        return {
            parent: self.edges[parent][node]
            for parent in self.parents(node)
        }

    def connect(self, node_a, node_b, edge: _Edge):
        self.edges[node_a][node_b] = edge
        self._parents[node_b].add(node_a)

    def sorted(self, entry_node=None):
        """Sort nodes topologically (breadth-first), starting with *entry_node*.

         In __ C __ X __ Out
          \__ D __ Y __/
        """
        if entry_node is None:
            entry_node = self.input_node

        queue = [entry_node]
        sorted_ = []
        while queue:
            node = queue.pop(0)
            sorted_.append(node)
            queue += [child for child in self.children(node) if
                      all(p in sorted_ for p in self.parents(child))]
        return sorted_

    def in_nodes_and_edges(self):
        for in_node in self.in_nodes:
            edge = self.edges[self.input_node][in_node]
            yield in_node, edge

    def out_nodes_and_edges(self):
        for out_node in self.out_nodes:
            edge = self.edges[out_node][self.output_node]
            yield out_node, edge

    def insert_graph(self, node_before, graph, node_after):
        graph = graph.copy()
        self.edges.update(graph.edges)

        for in_node, edge in graph.in_nodes_and_edges():
            self.connect(node_before, in_node, edge)

        del self.edges[graph.input_node]

        for out_node, edge in graph.out_nodes_and_edges():
            self.connect(out_node, node_after, edge)
            del self.edges[out_node][graph.output_node]

        self.update_parents()

    def replace_node(self, previous_node, new_node):
        self.edges[new_node] = self.edges[previous_node]
        del self.edges[previous_node]
        for child_dict in self.edges.values():
            try:
                del child_dict[new_node]
            except KeyError:
                pass
        self.update_parents()

    def replace_in_out(self):
        new_in = _Node()
        new_out = _Node()
        self.replace_node(self.input_node, new_in)
        self.replace_node(self.output_node, new_out)
        self.input_node = new_in
        self.output_node = new_out

    def convert_to_networkx(self):  # A: remove?
        """Convert Pipleline to Networkx graph

        Useful for drawing
        """
        networkx_graph = nx.DiGraph()
        for parent, children_dict in self.edges.items():
            networkx_graph.add_node(str(parent), node=parent)
            for child in children_dict:
                networkx_graph.add_node(str(child), node=child)
                edge = self.edges[parent][child]
                networkx_graph.add_edge(str(parent),
                                        str(child),
                                        label=str(edge))
        self.networkx_graph = networkx_graph

    def draw(self):  # A: remove?
        """Draw the pipeline using pygraphviz

        :return: Image
        """
        self.convert_to_networkx()
        dot = nx.nx_agraph.to_agraph(self.networkx_graph)
        dot.layout('dot')
        return Image(dot.draw(format='png', prog='dot'))


def _connect_graphs(from_graph, to_graph):
    # join the two graphs
    from_graph = from_graph.copy()
    to_graph = to_graph.copy()
    new_graph = _Graph(input_node=from_graph.input_node, output_node=to_graph.output_node)

    new_graph.edges.update(from_graph.edges)
    new_graph.edges.update(to_graph.edges)

    out_in = None if len(from_graph.out_nodes) == 1 else 0

    for out_node in from_graph.out_nodes:
        out_edge = new_graph.edges[out_node][from_graph.output_node]
        for in_node in to_graph.in_nodes:
            in_edge = to_graph.edges[to_graph.input_node][in_node]
            if out_edge.in_ is None or in_edge.out is None:
                new_graph.connect(out_node, in_node, _Edge(in_edge.out, out_in))
            elif in_edge.out == out_edge.in_:
                new_graph.connect(out_node, in_node, _Edge(None, None))
        # remove connections to old output node
        del new_graph.edges[out_node][from_graph.output_node]

    # remove connections to other's old input node
    del new_graph.edges[to_graph.input_node]
    new_graph.update_parents()
    return new_graph
