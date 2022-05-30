use crate::base::network::Network;
use crate::base::process::Process;
use crate::base::report::RunReport;
use crate::base::state::State;
use crate::base::stats::Stats;
use rand::rngs::SmallRng;
use rand::SeedableRng;

pub struct SimulatorConfig {
    bootstrap_steps: usize,
    window_steps: usize,
    max_windows: usize,
    termination_threshold: f32,
}

impl SimulatorConfig {

    pub fn new() -> Self {
        SimulatorConfig {
            bootstrap_steps: 300,
            window_steps: 200,
            max_windows: 1000,
            termination_threshold: 0.05,
        }
    }

    pub fn set_bootstrap_steps(&mut self, bootstrap_steps: usize) {
        self.bootstrap_steps = bootstrap_steps;
    }
    pub fn set_window_steps(&mut self, window_steps: usize) {
        self.window_steps = window_steps;
    }
    pub fn set_max_windows(&mut self, max_windows: usize) {
        self.max_windows = max_windows;
    }
    pub fn set_termination_threshold(&mut self, termination_threshold: f32) {
        self.termination_threshold = termination_threshold;
    }
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Simulator<'a, ProcessT: Process> {
    network: &'a Network,
    process: &'a ProcessT,
    rng: SmallRng,
    state: State<ProcessT>,

    stats: Vec<Stats<ProcessT>>,

    config: &'a SimulatorConfig,
}

impl<'a, ProcessT: Process> Simulator<'a, ProcessT> {
    pub fn new(config: &'a SimulatorConfig, network: &'a Network, process: &'a ProcessT) -> Self {
        let mut rng = SmallRng::seed_from_u64(0b1110110001110101011000111101); // Doc says that SmallRng should enough 1 in seed
        let state = process.make_initial_state(&mut rng, network);
        assert_eq!(state.node_count(), network.node_count());

        Simulator {
            network,
            process,
            rng,
            state,
            stats: vec![],
            config,
        }
    }

    pub(crate) fn step(&mut self, stats: &mut Stats<ProcessT>) {
        let graph = self.network.graph();
        let node_states = self.state.node_states();
        let mut idx = 0u32;
        let new_node_states: Vec<_> = node_states
            .iter()
            .zip(stats.node_stats_mut())
            .map(|(node_state, node_stats)| {
                let neighbors = graph
                    .neighbors(idx.into())
                    .map(|other| &node_states[other.index()]);
                idx += 1;
                self.process
                    .node_step(&mut self.rng, node_stats, node_state, neighbors)
            })
            .collect();
        self.state = State::new(new_node_states)
    }

    pub(crate) fn new_stats(&self) -> Stats<ProcessT> {
        Stats::new(self.network, self.config.window_steps)
    }

    pub fn stats(&self) -> &[Stats<ProcessT>] {
        &self.stats
    }

    pub fn report(&self) -> RunReport<ProcessT> {
        RunReport {
            process: self.process.get_report(self.network, &self.stats),
            network: self.network.name().to_string(),
            steps: self.stats.len() * self.config.window_steps,
        }
    }

    pub fn run(&mut self) -> bool {
        let mut stats = self.new_stats();
        for _i in 0..self.config.bootstrap_steps {
            self.step(&mut stats);
        }
        for _j in 0..self.config.max_windows {
            let mut stats = self.new_stats();
            for _i in 0..self.config.window_steps {
                self.step(&mut stats);
            }
            self.stats.push(stats);
            let metric = self.process.get_termination_metric(&self.stats);
            //dbg!(&metric);
            if metric
                .map(|x| x < self.config.termination_threshold)
                .unwrap_or(false)
            {
                return true;
            }
        }
        false
    }

    pub fn state(&mut self) -> &State<ProcessT> {
        &self.state
    }
}
