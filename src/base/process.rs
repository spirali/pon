use serde::Serialize;
use crate::base::network::Network;
use crate::base::state::State;
use crate::base::stats::Stats;

pub trait Process: Sized {
    type NodeStateT;
    type NodeStatsT: Default;
    type ReportT : Serialize;

    fn make_initial_state(&self, rng: &mut impl rand::Rng, network: &Network) -> State<Self>;

    //fn make_stats(&self, network: &Network) -> Self::StatsT;

    fn node_step<'a>(
        &'a self,
        rng: &mut impl rand::Rng,
        stats: &mut Self::NodeStatsT,
        node_state: &Self::NodeStateT,
        neighbors: impl Iterator<Item = &'a Self::NodeStateT>,
    ) -> Self::NodeStateT;

    fn get_termination_metric(&self, stats: &[Stats<Self>]) -> Option<f32>;

    fn get_report(&self, network: &Network, stats: &[Stats<Self>]) -> Self::ReportT;
}
