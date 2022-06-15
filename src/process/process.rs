use crate::games::game::ActionId;
use crate::process::network::Network;
use crate::process::state::State;
use serde::Serialize;

pub trait Process: Sized {
    type NodeStateT: Serialize;
    const ACTIONS: usize;

    fn make_initial_state(&self, rng: &mut impl rand::Rng, network: &Network) -> State<Self>;

    fn node_step(
        &self,
        rng: &mut impl rand::Rng,
        node_state: &Self::NodeStateT,
        last_action: ActionId,
        neighbors: impl Iterator<Item = ActionId>,
    ) -> (Self::NodeStateT, ActionId);

    fn configuration(&self) -> serde_json::Value;
}
