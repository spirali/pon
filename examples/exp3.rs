use ndarray::Array;
use ndarray::Axis;
use pon::env::streamer::Streamer;
use pon::games::chooser::SoftmaxSample;
use pon::games::counting::ActionCountingProcess;
use pon::games::game::{InitialAction, MatrixGame};
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

    let streamer = Streamer::new(Path::new("./hunt.data")).unwrap();
    scan! {
        //[stag_payoff = Array::linspace(3.0f32, 16.0, 40).to_vec()]
        let stag_payoff = 4.0;
        let payoffs = [[stag_payoff, 1.0], [3.0, 2.0]];

        [edge_prob = Array::linspace(0.1f64, 0.8, 10).to_vec()]
        [n_nodes = [32, 64]]
        [_replication = 0..16]
        let network = Network::random(&mut thread_rng(), n_nodes, edge_prob);

        let game = ActionCountingProcess::new(
            MatrixGame::new(payoffs, InitialAction::Uniform),
            SoftmaxSample::default(),
        );

        let mut rng = thread_rng();
        let mut simulator = Simulator::new(&config, Some(&mut rng), &network, &game);
        simulator.run();
        let report = simulator.report();
        let policy = report.avg_policy.mean_axis(Axis(0)).unwrap().into_raw_vec();
        let result = json!({
            "net": network.description(),
            "stag_payoff": stag_payoff,
            "policy": policy,
            "steps": report.steps});
        streamer.send(&result);
    }
}
