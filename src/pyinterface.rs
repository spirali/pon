use std::collections::HashMap;
use pyo3::prelude::*;
use std::sync::Arc;
use petgraph::algo::dominators::simple_fast;
use pyo3::exceptions::PyException;
use pyo3::types::PyDict;
use rand::{Rng, thread_rng};
use crate::matrixgame::MatrixGame;
use crate::simulation::{SimGraph, Simulation, SimulationResult};

#[pyclass]
#[derive(Clone)]
struct Graph {
    graph: Arc<SimGraph>
}

#[pymethods]
impl Graph {
    #[new]
    fn new(n_nodes: u32, edges: Vec<(u32, u32)>) -> PyResult<Self> {
        if let Some(e) = edges.iter().find(|x| x.0 >= n_nodes || x.1 >= n_nodes) {
            return Err(PyException::new_err(format!("Invalid edge {:?}", e)))
        };
        Ok(Graph {
            graph: Arc::new(SimGraph::from_edges(edges))
        })
    }
}



// [[f32; 2]; 2]
#[pyfunction]
fn run_matrix_game(payoff_matrix: [[f32; 2]; 2], graphs: Vec<Graph>, max_steps: usize, repeats: usize) -> PyResult<Vec<PyObject>> {
    let process = MatrixGame::new(payoff_matrix);
    let mut rng = thread_rng();
    let mut results: Vec<Vec<SimulationResult>> = graphs.iter().map(|_| Vec::with_capacity(repeats)).collect();
    for (graph_id, graph) in graphs.into_iter().enumerate() {
        for _ in 0..repeats {
            let mut simulation = Simulation::new(&graph.graph, &process, || rng.gen_range(0..2));
            let sim_result = simulation.run(&mut rng, max_steps);
            results[graph_id].push(sim_result);
        }
    }
    Python::with_gil(|py| {
        Ok(results.into_iter().map(|v| {
            Ok(v.into_iter().map(|r| {
                let mut dict = PyDict::new_bound(py);
                dict.set_item("history", r.history)?;
                PyResult::Ok(dict)
            }).collect::<PyResult<Vec<_>>>()?.to_object(py))
        }).collect::<PyResult<Vec<_>>>()?)
    })
}

/// A Python module implemented in Rust.
#[pymodule]
fn ponx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Graph>()?;
    m.add_function(wrap_pyfunction!(run_matrix_game, m)?)?;
    Ok(())
}
