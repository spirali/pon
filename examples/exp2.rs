use ndarray::{Array, Axis};
use pon::env::streamer::Streamer;
use pon::games::chooser::SamplingEpsilonError;
use pon::games::game::{InitialAction, MatrixGame};
use pon::games::regret::RegretMatchingProcess;
use pon::process::network::Network;
use pon::process::simulator::{Simulator, SimulatorConfig};
use pon::scan;
use rand::thread_rng;
use serde_json::json;
use std::path::Path;

pub fn main() {
    let mut config = SimulatorConfig::new();
    config.set_bootstrap_steps(20000);
    config.set_window_steps(20000);
    config.set_max_windows(500);
    config.set_termination_threshold(0.003);

    let payoffs = [[4.0, 1.0], [3.0, 2.0]];
    let streamer = Streamer::new(Path::new("./part3.data")).unwrap();
    scan! {
        [edge_prob = Array::linspace(0.05f64, 0.9, 5).to_vec()]
        // [n_nodes = 10..=10]
        [n_nodes = [32]]
        [_replication = 0..15]
        let network = Network::random(&mut thread_rng(), n_nodes, edge_prob);

        [epsilon = Array::linspace(0.4f32, 1.0, 5).to_vec()]
        let game = RegretMatchingProcess::new(
            MatrixGame::new(payoffs, InitialAction::Uniform),
            SamplingEpsilonError::new(epsilon),
        );

        [_replication = 0..10]
        let mut rng = thread_rng();
        let mut simulator = Simulator::new(&config, Some(&mut rng), &network, &game);
        simulator.run();
        let report = simulator.report();
        let policy = report.avg_policy.mean_axis(Axis(0)).unwrap().into_raw_vec();
        let result = json!({ "net": network.description(), "epsilon": epsilon, "policy": policy, "steps": report.steps});
        streamer.send(&result);
    }
    streamer.join();
}
