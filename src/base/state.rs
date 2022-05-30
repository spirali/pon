use crate::base::process::Process;

pub struct State<ProcessT: Process> {
    node_states: Vec<ProcessT::NodeStateT>,
}

impl<ProcessT: Process> State<ProcessT> {
    pub fn new(node_states: Vec<ProcessT::NodeStateT>) -> Self {
        Self { node_states }
    }

    pub fn node_states(&self) -> &[ProcessT::NodeStateT] {
        &self.node_states
    }
    pub fn node_count(&self) -> usize {
        self.node_states.len()
    }
}
