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
use std::fs;
use std::path::{Path, PathBuf};

fn scan_dir_for_graphs(path: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    Ok(fs::read_dir("../netprocess-data/v0.2/data-S")?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if path
                .to_str()
                .map(|s| s.ends_with(".edges.json"))
                .unwrap_or(false)
            {
                Some(path)
            } else {
                None
            }
        })
        .collect())
}

pub fn main() {
    let network_filenames =
        scan_dir_for_graphs(Path::new("../netprocess-data/v0.2/data-L")).unwrap();
    let mut config = SimulatorConfig::new();
    config.set_bootstrap_steps(20000);
    config.set_window_steps(20000);
    config.set_max_windows(1000);
    config.set_termination_threshold(0.003);

    let streamer = Streamer::new(Path::new("./hunt.data")).unwrap();

    scan! {
        //[stag_payoff = Array::linspace(3.0f32, 16.0, 40).to_vec()]
        let stag_payoff = 4.0;
        let payoffs = [[stag_payoff, 1.0], [3.0, 2.0]];


        [network_filename = &network_filenames]
        let network = Network::load_json(network_filename);

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
