use crate::base::network::Network;
use crate::base::process::Process;
use crate::base::state::State;
use crate::games::game::{ActionId, MatrixGame};
use rand::distributions::Bernoulli;
use rand::Rng;
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Debug, Serialize)]
pub struct PlayerState {
    action: ActionId,
}

#[derive(Debug)]
pub struct BestResponseProcess<const NACTIONS: usize> {
    dist_choose_best_resp: Bernoulli,
    prob_of_best_response: f64,
    game: MatrixGame<NACTIONS>,
}

impl<const ACTIONS: usize> BestResponseProcess<ACTIONS> {
    pub fn new(game: MatrixGame<ACTIONS>, prob_of_best_response: f64) -> Self {
        BestResponseProcess {
            dist_choose_best_resp: Bernoulli::new(prob_of_best_response).unwrap(),
            prob_of_best_response,
            game,
        }
    }
}

impl<const ACTIONS: usize> Process for BestResponseProcess<ACTIONS> {
    type NodeStateT = PlayerState;
    const ACTIONS: usize = ACTIONS;
    type CacheT = ();

    fn make_initial_state(&self, rng: &mut impl Rng, network: &Network) -> State<Self> {
        State::new_by(network, || PlayerState {
            action: self.game.make_initial_action(rng),
        })
    }

    fn init_cache(&self) -> Self::CacheT {}

    fn node_step<'a>(
        &'a self,
        rng: &mut impl Rng,
        _node_state: &PlayerState,
        neighbors: impl Iterator<Item = &'a PlayerState>,
        _cache: &mut (),
    ) -> (PlayerState, ActionId) {
        let action = if rng.sample(&self.dist_choose_best_resp) {
            let payoffs = self.game.payoffs_sums(neighbors.map(|s| s.action));
            payoffs.argmax() as ActionId
        } else {
            rng.gen_range(0..ACTIONS) as ActionId
        };
        (PlayerState { action }, action)
    }

    fn configuration(&self) -> Value {
        json!({
            "game": "br",
            "prob": self.prob_of_best_response
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::base::network::Network;
    use crate::base::simulator::{Simulator, SimulatorConfig};
    use crate::games::bestresp::BestResponseProcess;
    use crate::games::game::{InitialAction, MatrixGame};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_game_simple() {
        let game = BestResponseProcess::new(
            MatrixGame::new([[1.0, 0.5], [0.5, 0.0]], InitialAction::Const(1)),
            1.0,
        );
        let network = Network::line(2);
        let simulator = SimulatorConfig::new();
        let mut simulator = Simulator::new(&simulator, &network, &game);

        assert_eq!(simulator.state().node_states().len(), 2);
        for s in simulator.state().node_states() {
            assert_eq!(s.action, 1);
        }

        let mut stats = simulator.new_monitor();
        simulator.step(&mut stats, &mut ());

        assert_eq!(simulator.state().node_states().len(), 2);
        for s in simulator.state().node_states() {
            assert_eq!(s.action, 0);
        }
    }

    #[test]
    fn test_game_convergence() {
        let game = BestResponseProcess::new(
            MatrixGame::new([[0.0, 0.0], [0.0, 1.0]], InitialAction::Random),
            0.9,
        );
        let network = Network::grid(5, 5);
        let config = SimulatorConfig::new();
        let mut simulator = Simulator::new(&config, &network, &game);

        assert_eq!(simulator.state().node_states().len(), 25);
        assert!(simulator
            .state()
            .node_states()
            .iter()
            .any(|s| s.action == 0));
        assert!(simulator
            .state()
            .node_states()
            .iter()
            .any(|s| s.action == 1));

        simulator.run();

        let report = simulator.report();
        dbg!(&report);
        assert_eq!(report.steps, 1000);
        let avg_policy = &report.avg_policy;
        assert_eq!(avg_policy.len(), 2);
        assert_abs_diff_eq!(avg_policy[0] + avg_policy[1], 1.0, epsilon = 0.00001);
        assert_abs_diff_eq!(avg_policy[1], 0.95, epsilon = 0.01);
    }

    #[test]
    fn test_game_rock_paper_scissors() {
        let mut config = SimulatorConfig::new();
        config.set_termination_threshold(0.03);
        config.set_window_steps(1000);
        config.set_max_windows(100);

        let payoffs = [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]];
        let game =
            BestResponseProcess::<3>::new(MatrixGame::new(payoffs, InitialAction::Const(0)), 0.9);
        let network = Network::grid(5, 5);
        let mut simulator = Simulator::new(&config, &network, &game);
        simulator.run();
        let report = simulator.report();
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
