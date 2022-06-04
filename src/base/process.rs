use crate::base::network::Network;
use crate::base::state::State;
use crate::games::game::ActionId;

pub trait Process: Sized {
    type NodeStateT;
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
