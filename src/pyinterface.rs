use std::collections::HashMap;
use std::io::repeat;
use pyo3::prelude::*;
use std::sync::Arc;
use indicatif::{ProgressBar, ProgressIterator};
use petgraph::algo::dominators::simple_fast;
use pyo3::exceptions::PyException;
use pyo3::types::PyDict;
use rand::{Rng, thread_rng};
use crate::matrixgame::MatrixGame;
use crate::simulation::{RunConfig, SimGraph, Simulation, RunResult, EndReason};

#[pyclass]
#[derive(Clone)]
struct Graph {
    graph: Arc<SimGraph>
}

#[pymethods]
impl Graph {
    #[new]
    fn new(n_nodes: u32, edges: Vec<u32>) -> PyResult<Self> {
        if edges.len() % 2 != 0 {
            return Err(PyException::new_err(format!("Invalid number of edges: {}", edges.len())))
        }
        if let Some(e) = edges.iter().find(|x| **x >= n_nodes) {
            return Err(PyException::new_err(format!("Invalid edge {:?}", e)))
        };
        let mut graph = SimGraph::new_undirected();
        for _ in 0..n_nodes {
            graph.add_node(());
        }
        for edge in edges.chunks(2) {
            graph.add_edge(edge[0].into(), edge[1].into(), ());
        }
        Ok(Graph {
            graph: Arc::new(graph)
        })
    }

    fn edges(&self) -> PyResult<Vec<(u32, u32)>> {
        todo!()
    }
}

#[derive(FromPyObject)]
struct PyRunConfig {
    max_steps: usize,
    store_steps: usize,
}

impl PyRunConfig {
    fn into_config(self) -> RunConfig {
        RunConfig {
            max_steps: self.max_steps,
            store_steps: self.store_steps,
        }
    }
}


#[pyfunction]
fn run_matrix_game(payoff_matrix: [[f32; 2]; 2], graphs: Vec<Graph>, config: PyRunConfig, repeats: usize) -> PyResult<Vec<PyObject>> {
    let process = MatrixGame::new(payoff_matrix);
    let mut rng = thread_rng();
    let config = config.into_config();

    let mut work = Vec::with_capacity(graphs.len() * repeats);
    for (graph_idx, g) in graphs.iter().enumerate() {
        for _r in 0..repeats {
            work.push((graph_idx, g.graph.as_ref()))
        }
    }

    if work.is_empty() {
        return Ok(Vec::new())
    }

    let step = ((work.len() - 1) / 20) + 1;
    let mut next_step = step;
    let results: Vec<(usize, RunResult)> = work.iter().progress_count(work.len() as u64).enumerate().map(|(i, (graph_idx, graph))| {
        if i >= next_step {
            log::info!("Processed {}/{}", i + 1, work.len());
            next_step += step;
        }
        log::debug!("Processing {}", graph_idx);
        let mut simulation = Simulation::new(graph, &process, || rng.gen_range(0..2));
        let sim_result = simulation.run(&mut rng, &config);
        (*graph_idx, sim_result)
    }).collect();

    log::info!("Computation finished");

    Python::with_gil(|py| {
        Ok(results.into_iter().map(|(graph_idx, r)| {
                let mut dict = PyDict::new_bound(py);
                dict.set_item("graph_idx", graph_idx)?;
                dict.set_item("history", r.history)?;
                dict.set_item("means", r.means)?;
                dict.set_item("end_reason", match r.end_reason {
                    EndReason::Converged => 0,
                    EndReason::MeanCheck => 1,
                    EndReason::MaxStepsReached => 2,
                })?;
                PyResult::Ok(dict.to_object(py))
        }).collect::<PyResult<Vec<_>>>()?)
    })
}

/// A Python module implemented in Rust.
#[pymodule]
fn ponx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<Graph>()?;
    m.add_function(wrap_pyfunction!(run_matrix_game, m)?)?;
    Ok(())
}
