use pon::base::network::Network;
use pon::base::simulator::{Simulator, SimulatorConfig};
use pon::games::bestresp::BestResponseProcess;
use pon::games::game::{InitialAction, MatrixGame};

pub fn main() {
    let mut config = SimulatorConfig::new();
    config.set_window_steps(10000);
    config.set_termination_threshold(0.01);
    config.set_max_windows(100);
    //let payoffs = [[-1.0, -3.0], [0.0, -2.0]];
    let payoffs = [[4.0, 1.0], [3.0, 2.0]];

    let mut reports = Vec::new();
    for net_size in [2, 3, 4, 5, 6, 7] {
        let network = Network::grid(net_size, net_size);
        for prob in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] {
            let game = BestResponseProcess::new(
                MatrixGame::new(payoffs.clone(), InitialAction::Random),
                prob,
            );
            let mut simulator = Simulator::new(&config, &network, &game);
            simulator.run();
            let report = simulator.report();
            reports.push(report);
        }
    }

    println!("{}", serde_json::to_string(&reports).unwrap())
}
