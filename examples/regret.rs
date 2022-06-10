use ndarray::{Array, Axis};
use pon::env::streamer::{run_experiment, Streamer};
use pon::games::bestresp::BestResponseProcess;
use pon::games::chooser::EpsilonError;
use pon::games::game::{InitialAction, MatrixGame};
use pon::games::regmat::RegretMatchingProcess;
use pon::process::fixarray::FixArray;
use pon::process::network::Network;
use pon::process::simulator::{Simulator, SimulatorConfig};
use pon::scan;
use rand::thread_rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde_json::json;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub fn main() {
    let mut config = SimulatorConfig::new();
    config.set_bootstrap_steps(20000);
    config.set_window_steps(20000);
    config.set_max_windows(500);
    config.set_termination_threshold(0.003);

    let payoffs = [[4.0, 1.0], [3.0, 2.0]];
    let streamer = Streamer::new(Path::new("/tmp/data")).unwrap();
    scan! {
        [n = 7..=9]
        let network = Network::grid(n, n);

        [epsilon = Array::linspace(0.0f32, 1.0, 11).to_vec()]
        let game = RegretMatchingProcess::new(
            MatrixGame::new(payoffs, InitialAction::Uniform),
            EpsilonError::new(epsilon),
        );

        [_replication = 0..150]
        let mut rng = thread_rng();
        let mut simulator = Simulator::new(&config, Some(&mut rng), &network, &game);
        simulator.run();
        let report = simulator.report();
        let policy = report.avg_policy.mean_axis(Axis(0)).unwrap().into_raw_vec();
        let result = json!({ "network": network.name(), "epsilon": epsilon, "policy": policy, "steps": report.steps});
        streamer.send(&result);
    }
    streamer.join();
}
