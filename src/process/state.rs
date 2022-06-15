use crate::games::game::ActionId;
use crate::process::network::Network;
use crate::process::process::Process;

pub struct State<ProcessT: Process> {
    node_states: Vec<ProcessT::NodeStateT>,
    last_actions: Vec<ActionId>,
}

impl<ProcessT: Process> State<ProcessT> {
    pub fn new(node_states: Vec<ProcessT::NodeStateT>, last_actions: Vec<ActionId>) -> Self {
        Self {
            node_states,
            last_actions,
        }
    }

    pub fn new_by(
        network: &Network,
        mut init_fn: impl FnMut() -> (ProcessT::NodeStateT, ActionId),
    ) -> Self {
        let (states, actions) = network.graph().node_indices().map(|_| init_fn()).unzip();
        Self::new(states, actions)
    }

    pub fn node_states(&self) -> &[ProcessT::NodeStateT] {
        &self.node_states
    }

    pub fn last_actions(&self) -> &[ActionId] {
        &self.last_actions
    }

    pub fn node_count(&self) -> usize {
        self.node_states.len()
    }
}
