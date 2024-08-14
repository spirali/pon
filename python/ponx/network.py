
from . import ponx as ponx_rs


class Graph:

    def __init__(self, n_nodes, edges):
        self._graph = ponx_rs.Graph(n_nodes, edges)
        self.n_nodes = n_nodes
        self.n_edges = len(edges)

    def __repr__(self):
        return f"<Graph n_nodes={self.n_nodes} n_edges={self.n_edges}>"

    @staticmethod
    def circle(n) -> "Graph":
        edges = [(i, (i + 1) % n) for i in range(n)]
        return Graph(n, edges)
    