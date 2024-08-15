from typing_extensions import Sequence
from dataclasses import dataclass

from .network import Graph
from . import ponx as ponx_rs


@dataclass
class RunConfig:
    max_steps: int
    store_steps: int = 10

@dataclass
class RunResult:
    graph: Graph
    history: list[tuple[int, list[int]]]
    converged: bool

    @property
    def n_steps(self):
        return self.history[-1][0]


def _make_result(raw_result, graphs):
    graph = graphs[raw_result.pop("graph_idx")]
    return RunResult(graph=graph, **raw_result)


def run_matrix_game(payoffs: Sequence[Sequence[float]], graphs: Sequence[Graph] | Graph, config: RunConfig, repeats: int = 1):
    if isinstance(graphs, Graph):
        gs = [graphs._graph]
    else:
        gs = [g._graph for g in graphs]
    return [_make_result(r, graphs) for r in ponx_rs.run_matrix_game(payoffs, gs, config, repeats)]