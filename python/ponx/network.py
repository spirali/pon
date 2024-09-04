from . import ponx as ponx_rs
import tables
import numpy as np
import json

class Graph:
    def __init__(self, name, n_nodes, edges, metadata):
        self._graph = ponx_rs.Graph(n_nodes, edges)
        self.n_nodes = n_nodes
        self.n_edges = len(edges)
        self.name = name
        self.metadata = metadata

    def __repr__(self):
        return f"<Graph {self.name} n_nodes={self.n_nodes} n_edges={self.n_edges}>"

    @staticmethod
    def circle(n) -> "Graph":
        edges = []
        for i in range(n):
            edges.append(i)
            edges.append((i + 1) % n)
        return Graph(f"circle({n})", n, edges)

    @staticmethod
    def read(filename) -> "Graph":
        assert filename.endswith(".json")
        h5_name = filename[:-len(".json")] + ".h5"

        with open(filename, "r") as f:
            metadata = json.loads(f.read())

        f = tables.open_file(h5_name, "r")
        edges = []
        name_to_ids = {}
        for name in np.array(f.root.edges).flatten():
            node_id = name_to_ids.get(name)
            if node_id is None:
                node_id = len(name_to_ids)
                name_to_ids[name] = node_id
            edges.append(node_id)
        return Graph(filename, len(name_to_ids), edges, metadata)
