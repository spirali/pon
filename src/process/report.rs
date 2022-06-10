use ndarray::Array2;
use serde::Serialize;

#[derive(Serialize, Debug)]
pub struct RunReport {
    pub steps: usize,
    pub converged: bool,
    pub avg_policy: Array2<f32>,
}
