use crate::games::chooser::ActionChooser;
use crate::games::game::{ActionId, MatrixGame};
use crate::process::fixarray::FixArray;
use crate::process::network::Network;
use crate::process::process::Process;
use crate::process::state::State;
use rand::Rng;
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Debug, Serialize)]
pub struct PlayerState<const ACTIONS: usize> {
    action: ActionId,
    regret_sum: FixArray<ACTIONS>,
}

#[derive(Debug)]
pub struct RegretMatchingProcess<const ACTIONS: usize, ActionChooserT: ActionChooser<ACTIONS>> {
    game: MatrixGame<ACTIONS>,
    action_chooser: ActionChooserT,
}

impl<const ACTIONS: usize, ActionChooserT: ActionChooser<ACTIONS>>
    RegretMatchingProcess<ACTIONS, ActionChooserT>
{
    pub fn new(game: MatrixGame<ACTIONS>, action_chooser: ActionChooserT) -> Self {
        //let distribution = Bernoulli::new(p).unwrap()
        RegretMatchingProcess {
            game,
            action_chooser,
        }
    }
}

impl<const ACTIONS: usize, ActionChooserT: ActionChooser<ACTIONS>> Process
    for RegretMatchingProcess<ACTIONS, ActionChooserT>
{
    type NodeStateT = PlayerState<ACTIONS>;
    const ACTIONS: usize = ACTIONS;
    type CacheT = ();

    fn make_initial_state(&self, rng: &mut impl Rng, network: &Network) -> State<Self> {
        State::new_by(network, || PlayerState {
            action: self.game.make_initial_action(rng),
            regret_sum: FixArray::default(),
        })
    }

    fn init_cache(&self) -> Self::CacheT {}

    fn node_step<'a>(
        &'a self,
        rng: &mut impl Rng,
        node_state: &PlayerState<ACTIONS>,
        neighbors: impl Iterator<Item = &'a PlayerState<ACTIONS>>,
        _cache: &mut (),
    ) -> (PlayerState<ACTIONS>, ActionId) {
        let payoffs = self.game.payoffs_sums(neighbors.map(|s| s.action));
        let regret = payoffs.sub_scalar(payoffs.get(node_state.action));
        let regret_sum = node_state.regret_sum.add(&regret);
        let clamped = regret_sum.clamp_negatives();
        //let policy_sum = node_state.policy_sum.add(&clamped.normalize_to_policy());

        /*let action = if rng.gen_bool(0.9) {
            clamped.sample_index(rng)
        } else {
            rng.gen_range(0..ACTIONS)
        };*/
        let action = self.action_chooser.choose_action(rng, clamped);
        //println!("POL {} {}", policy_sum, policy);
        /*println!(
            "Regret {} reg={} r_sum={} p_sum={}",
            node_state.action, regret, regret_sum, policy_sum
        );*/
        (PlayerState { action, regret_sum }, action)
    }

    fn configuration(&self) -> Value {
        json!({
            "game": "rm",
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::games::bestresp::BestResponseProcess;
    use crate::games::chooser::DirectChooser;
    use crate::games::game::{InitialAction, MatrixGame};
    use crate::games::regmat::RegretMatchingProcess;
    use crate::process::network::Network;
    use crate::process::simulator::{Simulator, SimulatorConfig};
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;
    use std::path::Path;

    #[test]
    fn test_regret_matching_game_rock_paper_scissors() {
        let mut config = SimulatorConfig::new();
        config.set_termination_threshold(0.003);
        config.set_window_steps(10000);
        config.set_max_windows(100);
        config.set_trace_path(Path::new("/tmp/p"));
        config.set_report_state_step(200);

        let payoffs = [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]];
        let game = RegretMatchingProcess::new(
            MatrixGame::new(payoffs, InitialAction::Const(0)),
            DirectChooser::new(),
        );
        //let network = Network::grid(5, 5);
        let network = Network::line(2);
        let mut simulator = Simulator::new(&config, None, &network, &game);
        simulator.run();
        let report = simulator.report();

        dbg!(&report);
        let avg_policy = &report.avg_policy;
        assert_eq!(avg_policy.len(), 3);
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
