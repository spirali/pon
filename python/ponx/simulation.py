from typing_extensions import Sequence
from dataclasses import dataclass
import enum

from .network import Graph
from . import ponx as ponx_rs


@dataclass
class RunConfig:
    max_steps: int
    store_steps: int = 10


@enum.unique
class EndReason(enum.IntEnum):
    Converged = 0
    MeanCheck = 1
    MaxLimitReached = 2


@dataclass
class RunResult:
    graph: Graph
    history: list[tuple[int, list[int]]]
    means: list[float]
    end_reason: EndReason

    def __post_init__(self):
        self.end_reason = EndReason(self.end_reason)

    @property
    def n_steps(self):
        return self.history[-1][0]


def _make_result(raw_result, graphs):
    graph = graphs[raw_result.pop("graph_idx")]
    return RunResult(graph=graph, **raw_result)


def run_matrix_game(
    payoffs: Sequence[Sequence[float]],
    graphs: Sequence[Graph] | Graph,
    config: RunConfig,
    repeats: int = 1,
):
    if isinstance(graphs, Graph):
        gs = [graphs._graph]
    else:
        gs = [g._graph for g in graphs]
    return [
        _make_result(r, graphs)
        for r in ponx_rs.run_matrix_game(payoffs, gs, config, repeats)
    ]
