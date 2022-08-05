use ndarray::{Array, Axis};
use pon::env::streamer::Streamer;
use pon::games::chooser::SamplingEpsilonError;
use pon::games::chooser::SoftmaxSample;
use pon::games::game::{InitialAction, MatrixGame};
use pon::games::regret::RegretMatchingProcess;
use pon::process::fixarray::FloatArray;
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
    let streamer = Streamer::new(Path::new("./e1.data")).unwrap();
    scan! {
    /*        [edge_prob = Array::linspace(0.05f64, 0.9, 5).to_vec()]
            // [n_nodes = 10..=10]
            [n_nodes = [32]]
            [_replication = 0..15]
            let network = Network::random(&mut thread_rng(), n_nodes, edge_prob);
     */
            let network = Network::line(2);

            [init_dist = Array::linspace(0.0f64, 1.0, 20).to_vec()]
            let init_dist = init_dist as f32;
            let game = RegretMatchingProcess::new(
                MatrixGame::new(payoffs, InitialAction::Distribution(FloatArray::from([init_dist, 1.0 - init_dist]))),
                SoftmaxSample::default(),
            );

            [_replication = 0..1000]
            let mut rng = thread_rng();
            let mut simulator = Simulator::new(&config, Some(&mut rng), &network, &game);
            simulator.run();
            let report = simulator.report();
            let policy = report.avg_policy.into_raw_vec();
            //dbg!(&policy);
            let result = json!({ "net": network.description(), "init_dist": init_dist, "policy": policy, "steps": report.steps});
            streamer.send(&result);
        }
}
