use pon::base::network::Network;
use pon::base::simulator::{Simulator, SimulatorConfig};
use pon::games::bestresp::BestResponseProcess;
use pon::games::game::{InitialAction, MatrixGame};
use pon::games::regmat::RegretMatchingProcess;
use std::path::Path;

pub fn main() {
    let mut config = SimulatorConfig::new();
    config.set_termination_threshold(0.003);
    config.set_window_steps(10000);
    config.set_max_windows(10000);
    config.set_termination_threshold(0.00001);
    config.set_trace_path(Path::new("/tmp/p"));
    //config.set_report_state_step(200);

    let payoffs = [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]];

    let game = RegretMatchingProcess::<3>::new(MatrixGame::new(payoffs, InitialAction::Const(0)));
    /*let game =
    BestResponseProcess::new(MatrixGame::new(payoffs.clone(), InitialAction::Random), 0.9);*/

    //let network = Network::grid(5, 5);
    let network = Network::line(2);
    let mut simulator = Simulator::new(&config, &network, &game);
    simulator.run();
    let report = simulator.report();
    dbg!(&report);
    let avg_policy = &report.avg_policy;
    assert_eq!(avg_policy.len(), 3);
}
