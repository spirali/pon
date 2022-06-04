use crate::base::monitor::Monitor;
use crate::base::network::Network;
use crate::base::process::Process;
use crate::base::report::RunReport;
use crate::base::state::State;
use ndarray::{stack, Array1, Axis};
use ordered_float::OrderedFloat;
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

    avg_policies: Vec<Array1<f32>>,
    converged: bool,

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
            avg_policies: vec![],
            converged: false,
            config,
        }
    }

    pub(crate) fn step(&mut self, stats: &mut Monitor, cache: &mut ProcessT::CacheT) {
        let graph = self.network.graph();
        let node_states = self.state.node_states();
        let mut idx = 0u32;
        stats.new_step();
        let new_node_states: Vec<_> = node_states
            .iter()
            .map(|node_state| {
                let neighbors = graph
                    .neighbors(idx.into())
                    .map(|other| &node_states[other.index()]);
                let (new_state, action) =
                    self.process
                        .node_step(&mut self.rng, node_state, neighbors, cache);
                stats.record_action(idx, action);
                idx += 1;
                new_state
            })
            .collect();
        self.state = State::new(new_node_states)
    }

    pub(crate) fn new_stats(&self) -> Monitor {
        Monitor::new(self.network, ProcessT::ACTIONS)
    }

    pub fn report(&self) -> RunReport {
        RunReport {
            network: self.network.name().to_string(),
            process: self.process.configuration(),
            steps: self.avg_policies.len() * self.config.window_steps,
            converged: self.converged,
            avg_policy: self.avg_policies.last().unwrap().to_vec(),
        }
    }

    pub fn run(&mut self) -> bool {
        let mut stats = self.new_stats();
        let mut cache = self.process.init_cache();
        for _i in 0..self.config.bootstrap_steps {
            self.step(&mut stats, &mut cache);
        }
        drop(stats);
        for _j in 0..self.config.max_windows {
            let mut stats = self.new_stats();
            for _i in 0..self.config.window_steps {
                self.step(&mut stats, &mut cache);
            }
            self.avg_policies.push(stats.avg_policy());
            let metric = self.get_termination_metric();
            //dbg!(&metric);
            if metric
                .map(|x| x < self.config.termination_threshold)
                .unwrap_or(false)
            {
                self.converged = true;
                return true;
            }
        }
        false
    }

    fn get_termination_metric(&self) -> Option<f32> {
        const SUFFIX_SIZE: usize = 4;
        if self.avg_policies.len() < SUFFIX_SIZE + 1 {
            return None;
        }
        let mut it = self.avg_policies.iter().rev();
        let last: &Array1<f32> = it.next().unwrap();
        let prevs: Vec<_> = it.take(SUFFIX_SIZE).map(|x| x.view()).collect();
        let prev_avg_policy: Array1<f32> =
            stack(Axis(0), &prevs).unwrap().mean_axis(Axis(0)).unwrap();
        let max_diff = (prev_avg_policy - last)
            .iter()
            .map(|x| OrderedFloat::from(x.abs()))
            .max()
            .unwrap();
        Some(max_diff.into_inner())
    }

    pub fn state(&mut self) -> &State<ProcessT> {
        &self.state
    }
}
