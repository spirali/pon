use serde::Serialize;
use serde_json::Value;

#[derive(Serialize, Debug)]
pub struct RunReport {
    pub network: String,
    pub process: Value,
    pub steps: usize,
    pub converged: bool,
    pub avg_policy: Vec<f32>,
}
