use crate::base::network::Network;
use crate::base::process::Process;

#[derive(Debug)]
pub struct Stats<ProcessT: Process> {
    node_stats: Vec<ProcessT::NodeStatsT>,
    window_size: usize,
}

impl<ProcessT: Process> Stats<ProcessT> {
    pub fn new(network: &Network, window_size: usize) -> Self {
        let mut node_stats = Vec::new();
        node_stats.resize_with(network.node_count(), ProcessT::NodeStatsT::default);
        Stats {
            node_stats,
            window_size,
        }
    }

    pub fn from_vec(node_stats: Vec<ProcessT::NodeStatsT>, window_size: usize) -> Self {
        Stats {
            node_stats,
            window_size,
        }
    }

    pub fn node_stats_mut(&mut self) -> &mut [ProcessT::NodeStatsT] {
        &mut self.node_stats
    }

    pub fn node_stats(&self) -> &[ProcessT::NodeStatsT] {
        &self.node_stats
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }
}
