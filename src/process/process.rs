use crate::games::game::ActionId;
use crate::process::network::Network;
use crate::process::state::State;
use serde::Serialize;

pub trait Process: Sized {
    type NodeStateT: Serialize;
    const ACTIONS: usize;
    type CacheT;

    fn make_initial_state(&self, rng: &mut impl rand::Rng, network: &Network) -> State<Self>;

    fn init_cache(&self) -> Self::CacheT;

    fn node_step<'a>(
        &'a self,
        rng: &mut impl rand::Rng,
        node_state: &Self::NodeStateT,
        neighbors: impl Iterator<Item = &'a Self::NodeStateT>,
        cache: &mut Self::CacheT,
    ) -> (Self::NodeStateT, ActionId);

    fn configuration(&self) -> serde_json::Value;
}
