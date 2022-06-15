use crate::games::chooser::ActionChooser;
use crate::games::game::{ActionId, MatrixGame};
use crate::process::fixarray::IntArray;
use crate::process::network::Network;
use crate::process::process::Process;
use crate::process::state::State;
use rand::Rng;
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Debug, Serialize)]
pub struct PlayerState<const ACTIONS: usize> {
    action_counts: IntArray<ACTIONS>,
}

#[derive(Debug)]
pub struct ActionCountingProcess<const ACTIONS: usize, ActionChooserT: ActionChooser<ACTIONS>> {
    game: MatrixGame<ACTIONS>,
    action_chooser: ActionChooserT,
}

impl<const ACTIONS: usize, ActionChooserT: ActionChooser<ACTIONS>>
    ActionCountingProcess<ACTIONS, ActionChooserT>
{
    pub fn new(game: MatrixGame<ACTIONS>, action_chooser: ActionChooserT) -> Self {
        ActionCountingProcess {
            game,
            action_chooser,
        }
    }
}

impl<const ACTIONS: usize, ActionChooserT: ActionChooser<ACTIONS>> Process
    for ActionCountingProcess<ACTIONS, ActionChooserT>
{
    type NodeStateT = PlayerState<ACTIONS>;
    const ACTIONS: usize = ACTIONS;

    fn make_initial_state(&self, rng: &mut impl Rng, network: &Network) -> State<Self> {
        State::new_by(network, || {
            (
                PlayerState::<ACTIONS> {
                    action_counts: IntArray::default(),
                },
                self.game.make_initial_action(rng),
            )
        })
    }

    fn node_step(
        &self,
        rng: &mut impl Rng,
        node_state: &PlayerState<ACTIONS>,
        _last_action: ActionId,
        neighbors: impl Iterator<Item = ActionId>,
    ) -> (PlayerState<ACTIONS>, ActionId) {
        let mut counts = node_state.action_counts.clone();
        for action in neighbors {
            *counts.get_mut(action) += 1;
        }
        let probs = counts.as_float().normalize();
        let payoffs = self.game.expect_payoffs(probs);
        let action = self.action_chooser.choose_action(rng, payoffs);
        (
            PlayerState {
                action_counts: counts,
            },
            action,
        )
    }

    fn configuration(&self) -> Value {
        json!({
            "game": "br",
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::games::chooser::BestResponseEpsilonError;
    use crate::games::counting::ActionCountingProcess;
    use crate::games::game::{InitialAction, MatrixGame};
    use crate::process::network::Network;
    use crate::process::simulator::{Simulator, SimulatorConfig};
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;

    #[test]
    fn test_game_simple() {
        let game = ActionCountingProcess::new(
            MatrixGame::new([[1.0, 0.5], [0.5, 0.0]], InitialAction::Const(1)),
            BestResponseEpsilonError::new(0.0),
        );
        let network = Network::line(2);
        let simulator = SimulatorConfig::new();
        let mut simulator = Simulator::new(&simulator, None, &network, &game);

        assert_eq!(simulator.state().node_states().len(), 2);
        for action in simulator.state().last_actions() {
            assert_eq!(*action, 1);
        }

        simulator.step();

        assert_eq!(simulator.state().node_states().len(), 2);
        for action in simulator.state().last_actions() {
            assert_eq!(*action, 0);
        }
    }

    #[test]
    fn test_game_convergence() {
        let game = ActionCountingProcess::new(
            MatrixGame::new([[0.0, 0.0], [0.0, 1.0]], InitialAction::Uniform),
            BestResponseEpsilonError::new(0.1),
        );
        let network = Network::grid(5, 5);
        let config = SimulatorConfig::new();
        let mut simulator = Simulator::new(&config, None, &network, &game);

        assert_eq!(simulator.state().node_states().len(), 25);
        assert!(simulator.state().last_actions().iter().any(|a| *a == 0));
        assert!(simulator.state().last_actions().iter().any(|a| *a == 1));

        simulator.run();

        let report = simulator.report();
        dbg!(&report);
        let avg_policy = report.avg_policy;
        /*assert_eq!(avg_policy.len(), 2);*/
        assert_abs_diff_eq!(
            (&avg_policy.column(0) + &avg_policy.column(1))
                .mean()
                .unwrap(),
            1.0,
            epsilon = 0.00001
        );
        for i in 0..25 {
            assert_abs_diff_eq!(avg_policy[(i, 1)], 0.95, epsilon = 0.01);
        }
    }

    #[test]
    fn test_game_rock_paper_scissors() {
        let mut config = SimulatorConfig::new();
        config.set_termination_threshold(0.03);
        config.set_window_steps(1000);
        config.set_max_windows(100);

        let payoffs = [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]];
        let game = ActionCountingProcess::<3, _>::new(
            MatrixGame::new(payoffs, InitialAction::Const(0)),
            BestResponseEpsilonError::new(0.1),
        );
        let network = Network::grid(5, 5);
        let mut simulator = Simulator::new(&config, None, &network, &game);
        simulator.run();
        let report = simulator.report();
        let avg_policy = report.avg_policy;
        //assert_eq!(avg_policy.len(), 3);
        assert_abs_diff_eq!(
            (&avg_policy.column(0) + &avg_policy.column(1) + &avg_policy.column(2))
                .mean()
                .unwrap(),
            1.0,
            epsilon = 0.00001
        );
        let a = avg_policy.mean_axis(Axis(0)).unwrap();
        assert_abs_diff_eq!(a[0], 0.33, epsilon = 0.1);
        assert_abs_diff_eq!(a[1], 0.33, epsilon = 0.1);
        assert_abs_diff_eq!(a[2], 0.33, epsilon = 0.1);
    }
}
