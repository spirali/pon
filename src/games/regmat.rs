use crate::base::fixarray::FixArray;
use crate::base::network::Network;
use crate::base::process::Process;
use crate::base::state::State;
use crate::games::game::{ActionId, MatrixGame};
use rand::distributions::{Bernoulli, WeightedIndex};
use rand::Rng;
use serde_json::{json, Value};

#[derive(Debug)]
pub struct PlayerState<const ACTIONS: usize> {
    action: ActionId,
    regret_sum: FixArray<ACTIONS>,
    policy_sum: FixArray<ACTIONS>,
}

#[derive(Debug)]
pub struct RegretMatchingProcess<const ACTIONS: usize> {
    game: MatrixGame<ACTIONS>,
}

impl<const ACTIONS: usize> RegretMatchingProcess<ACTIONS> {
    pub fn new(game: MatrixGame<ACTIONS>) -> Self {
        RegretMatchingProcess { game }
    }
}

impl<const ACTIONS: usize> Process for RegretMatchingProcess<ACTIONS> {
    type NodeStateT = PlayerState<ACTIONS>;
    const ACTIONS: usize = ACTIONS;
    type CacheT = ();

    fn make_initial_state(&self, rng: &mut impl Rng, network: &Network) -> State<Self> {
        State::new_by(&network, || PlayerState {
            action: self.game.make_initial_action(rng),
            regret_sum: FixArray::default(),
            policy_sum: FixArray::default(),
        })
    }

    fn init_cache(&self) -> Self::CacheT {
        ()
    }

    fn node_step<'a>(
        &'a self,
        rng: &mut impl Rng,
        node_state: &PlayerState<ACTIONS>,
        neighbors: impl Iterator<Item = &'a PlayerState<ACTIONS>>,
        _cache: &mut (),
    ) -> (PlayerState<ACTIONS>, ActionId) {
        let payoffs = self.game.payoffs_sums(neighbors.map(|s| s.action));
        let regret_sum = node_state.regret_sum.add(&payoffs);
        let policy = regret_sum.clamp_negatives().normalize();
        let policy_sum = node_state.policy_sum.add(&policy);
        let action = policy_sum.sample_index(rng);
        /*println!(
            "Regret {} -> {}; {} -> {}",
            node_state.regret_sum, regret_sum, node_state.policy_sum, policy_sum
        );*/
        (
            PlayerState {
                action,
                regret_sum,
                policy_sum,
            },
            action,
        )
    }

    fn configuration(&self) -> Value {
        json!({
            "game": "rm",
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::base::network::Network;
    use crate::base::simulator::{Simulator, SimulatorConfig};
    use crate::games::bestresp::BestResponseProcess;
    use crate::games::game::{InitialAction, MatrixGame};
    use crate::games::regmat::RegretMatchingProcess;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_regret_matching_game_rock_paper_scissors() {
        let mut config = SimulatorConfig::new();
        config.set_termination_threshold(0.03);
        config.set_window_steps(1000);
        config.set_max_windows(100);

        let payoffs = [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]];
        let game =
            RegretMatchingProcess::<3>::new(MatrixGame::new(payoffs, InitialAction::Const(1)));
        //let network = Network::grid(5, 5);
        let network = Network::line(2);
        let mut simulator = Simulator::new(&config, &network, &game);
        simulator.run();
        let report = simulator.report();

        dbg!(&report);
        let avg_policy = &report.avg_policy;
        assert_eq!(avg_policy.len(), 3);
        assert_abs_diff_eq!(
            avg_policy[0] + avg_policy[1] + avg_policy[2],
            1.0,
            epsilon = 0.00001
        );
        assert_abs_diff_eq!(avg_policy[0], 0.33, epsilon = 0.1);
        assert_abs_diff_eq!(avg_policy[1], 0.33, epsilon = 0.1);
        assert_abs_diff_eq!(avg_policy[2], 0.33, epsilon = 0.1);
    }
}
