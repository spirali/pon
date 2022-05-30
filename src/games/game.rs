use crate::base::network::Network;
use crate::base::process::Process;
use crate::base::state::State;
use crate::base::stats::Stats;
use ndarray::{Array2, Axis};
use ordered_float::NotNan;
use rand::distributions::Bernoulli;
use rand::Rng;
use serde::Serialize;

type ActionId = u32;

#[derive(Debug, Clone)]
pub struct PlayerStats<const NACTIONS: usize> {
    action_counts: [u32; NACTIONS],
}

#[derive(Debug, Clone, Serialize)]
pub struct GameReport {
    pub prob_of_best_response: f64,
    pub avg_policy: Vec<f32>,
}

impl<const NACTIONS: usize> Default for PlayerStats<NACTIONS> {
    fn default() -> Self {
        PlayerStats {
            action_counts: [0; NACTIONS],
        }
    }
}

#[derive(Debug)]
pub struct PlayerState {
    action: ActionId,
}

#[derive(Debug)]
pub enum InitialAction {
    Const(ActionId),
    Random,
}

#[derive(Debug)]
pub struct MatrixGame<const NACTIONS: usize> {
    payoff_matrix: [[f32; NACTIONS]; NACTIONS],
    dist_choose_best_resp: Bernoulli,
    init_action: InitialAction,
    prob_of_best_response: f64,
}

impl<const NACTIONS: usize> MatrixGame<NACTIONS> {
    pub fn new(
        payoff_matrix: [[f32; NACTIONS]; NACTIONS],
        prob_of_best_response: f64,
        init_action: InitialAction,
    ) -> Self {
        MatrixGame {
            payoff_matrix,
            dist_choose_best_resp: Bernoulli::new(prob_of_best_response).unwrap(),
            init_action,
            prob_of_best_response
        }
    }
    /*pub fn action_count(&self) -> ActionId {
        //self.payoff_matrix.nrows() as ActionId
        NACTIONS as ActionId
    }*/
}

impl<const NACTIONS: usize> Process for MatrixGame<NACTIONS> {
    type NodeStateT = PlayerState;
    type NodeStatsT = PlayerStats<NACTIONS>;
    type ReportT = GameReport;

    fn make_initial_state(&self, rng: &mut impl rand::Rng, network: &Network) -> State<Self> {
        let n_actions = NACTIONS as ActionId;
        match self.init_action {
            InitialAction::Const(action) => State::new(
                (0..network.node_count())
                    .map(|_| PlayerState { action })
                    .collect(),
            ),
            InitialAction::Random => State::new(
                (0..network.node_count())
                    .map(|_| PlayerState {
                        action: rng.gen_range(0..n_actions),
                    })
                    .collect(),
            ),
        }
    }

    fn node_step<'a>(
        &'a self,
        rng: &mut impl Rng,
        node_stats: &mut PlayerStats<NACTIONS>,
        _node_state: &PlayerState,
        neighbors: impl Iterator<Item = &'a PlayerState>,
    ) -> PlayerState {
        //let payoffs = self.payoff_matrix.row(node_state.action as usize);
        let mut payoffs: [f32; NACTIONS] = [0.0; NACTIONS];
        let mut count = 0u32;
        neighbors.for_each(|s| {
            for (i, p) in payoffs.iter_mut().enumerate() {
                *p += self.payoff_matrix[i][s.action as usize];
            }
            count += 1;
        });

        let action = if rng.sample(&self.dist_choose_best_resp) {
            let (max_idx, _) = payoffs.iter().enumerate().fold(
                (0, payoffs[0]),
                |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                },
            );
            max_idx as ActionId
        } else {
            rng.gen_range(0..NACTIONS) as ActionId
        };

        node_stats.action_counts[action as usize] += 1;
        PlayerState { action }
    }

    fn get_termination_metric(&self, all_stats: &[Stats<Self>]) -> Option<f32> {
        const SUFFIX_SIZE: usize = 5;
        if all_stats.len() < SUFFIX_SIZE + 1 {
            return None;
        }
        let it = all_stats.iter().rev();
        let n_nodes = all_stats.last().unwrap().node_stats().len();
        let avg_policy = policies_from_stats(it.skip(1).take(SUFFIX_SIZE), n_nodes);
        let last_stats = all_stats.last().unwrap();
        let last_window_size = last_stats.window_size() as f32;
        let max_diff = avg_policy
            .iter()
            .zip(last_stats.node_stats())
            .map(|(sm, ns)| {
                sm.iter()
                    .zip(ns.action_counts)
                    .map(|(m, n)| NotNan::new(*m - n as f32 / last_window_size).unwrap())
                    .max()
                    .unwrap()
            })
            .max()
            .unwrap();
        Some(max_diff.into_inner())
    }

    fn get_report(&self, network: &Network, stats: &[Stats<Self>]) -> GameReport {
        let len = stats.len();
        // Make policy from last 3 windows
        let policies = policies_from_stats(stats[len - 3..len].iter(), network.node_count());
        let avgs = policies_into_array(&policies).mean_axis(Axis(0)).unwrap();
        GameReport {
            prob_of_best_response: self.prob_of_best_response,
            avg_policy: avgs.to_vec(),
        }
    }
}

fn policies_into_array<const NACTIONS: usize>(policies: &[[f32; NACTIONS]]) -> Array2<f32> {
    let mut array = Array2::<f32>::zeros((policies.len(), NACTIONS));
    for (row, row_values) in policies.iter().enumerate() {
        for (column, value) in row_values.iter().enumerate() {
            array[(row, column)] = *value;
        }
    }
    array
}

fn policies_from_stats<'a, const NACTIONS: usize>(
    stats_it: impl Iterator<Item = &'a Stats<MatrixGame<NACTIONS>>>,
    n_nodes: usize,
) -> Vec<[f32; NACTIONS]> {
    let mut result = vec![[0f32; NACTIONS]; n_nodes];
    let mut window_size = 0;
    for s in stats_it {
        for (sm, ns) in result.iter_mut().zip(s.node_stats()) {
            for (m, n) in sm.iter_mut().zip(ns.action_counts) {
                *m += n as f32;
            }
        }
        window_size += s.window_size();
    }
    for sm in result.iter_mut() {
        for m in sm.iter_mut() {
            *m /= window_size as f32;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use crate::base::network::Network;
    use crate::base::simulator::{Simulator, SimulatorConfig};
    use crate::games::game::{InitialAction, MatrixGame};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_game_simple() {
        let game = MatrixGame::<2>::new([[1.0, 0.5], [0.5, 0.0]], 1.0, InitialAction::Const(1));
        let network = Network::line(2);
        let simulator = SimulatorConfig::new();
        let mut simulator = Simulator::new(&simulator, &network, &game);

        assert_eq!(simulator.state().node_states().len(), 2);
        for s in simulator.state().node_states() {
            assert_eq!(s.action, 1);
        }

        let mut stats = simulator.new_stats();
        simulator.step(&mut stats);

        assert_eq!(simulator.state().node_states().len(), 2);
        for s in simulator.state().node_states() {
            assert_eq!(s.action, 0);
        }
    }

    #[test]
    fn test_game_convergence() {
        let game = MatrixGame::<2>::new([[0.0, 0.0], [0.0, 1.0]], 0.9, InitialAction::Random);
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

        let size = simulator.stats().len();
        assert_eq!(size, 6);

        let report = simulator.report();
        let avg_policy = &report.process.avg_policy;
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
        let game = MatrixGame::<3>::new(payoffs, 0.9, InitialAction::Random);
        let network = Network::grid(5, 5);
        let mut simulator = Simulator::new(&config, &network, &game);
        simulator.run();
        let report = simulator.report();
        let avg_policy = &report.process.avg_policy;
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
