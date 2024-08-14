from typing_extensions import Sequence

from .network import Graph
from . import ponx as ponx_rs


def run_matrix_game(payoffs: Sequence[Sequence[float]], graphs: Sequence[Graph] | Graph, max_steps: int, repeats: int = 1):
    if isinstance(graphs, Graph):
        gs = [graphs._graph]
        unwrap = True
    else:
        gs = [g._graph for g in graphs]
        unwrap = False
    result = ponx_rs.run_matrix_game(payoffs, gs, max_steps, repeats)
    if unwrap:
        return result[0]
    else:
        return result