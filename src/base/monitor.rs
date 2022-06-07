use crate::base::network::Network;
use ndarray::{Array1, Array2, Axis};

#[derive(Debug)]
pub struct Monitor {
    action_counts: Array2<u32>,
    steps: usize,
}

impl Monitor {
    pub fn new(network: &Network, action_count: usize) -> Self {
        Monitor {
            action_counts: Array2::zeros((network.node_count(), action_count)),
            steps: 0,
        }
    }

    pub fn counts(&self) -> &Array2<u32> {
        &self.action_counts
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn record_action(&mut self, index: u32, action: usize) {
        self.action_counts[(index as usize, action)] += 1;
    }

    pub fn new_step(&mut self) {
        self.steps += 1;
    }

    pub fn avg_policy(&self) -> Array1<f32> {
        let f = (self.steps * self.action_counts.nrows()) as f32;
        self.action_counts.sum_axis(Axis(0)).mapv(|x| x as f32 / f)
    }
}
