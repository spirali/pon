use crate::base::monitor::Monitor;
use crate::base::network::Network;
use crate::base::process::Process;
use crate::base::report::RunReport;
use crate::base::state::State;
use ndarray::{stack, Array1, Axis};
use ordered_float::OrderedFloat;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct SimulatorConfig {
    bootstrap_steps: usize,
    window_steps: usize,
    max_windows: usize,
    termination_threshold: f32,
    trace_path: Option<PathBuf>,
    report_state_step: usize,
}

#[derive(Serialize)]
#[serde(tag = "evt")]
pub enum TraceFrame<'a, ProcessT: Process> {
    State(StateTraceFrame<'a, ProcessT::NodeStateT>),
    Window(WindowTraceFrame<'a>),
}

#[derive(Serialize)]
pub struct StateTraceFrame<'a, NodeStateT: Serialize> {
    step: usize,
    states: &'a [NodeStateT],
}

#[derive(Serialize)]
pub struct WindowTraceFrame<'a> {
    step: usize,
    actions: usize,
    counts: &'a [u32],
}

impl SimulatorConfig {
    pub fn new() -> Self {
        SimulatorConfig {
            bootstrap_steps: 300,
            window_steps: 200,
            max_windows: 1000,
            termination_threshold: 0.05,
            trace_path: None,
            report_state_step: 0,
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
    pub fn set_trace_path(&mut self, trace_path: &Path) {
        self.trace_path = Some(trace_path.into());
    }
    pub fn set_report_state_step(&mut self, report_step: usize) {
        self.report_state_step = report_step;
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
    step: usize,

    config: &'a SimulatorConfig,
    trace_file: Option<BufWriter<File>>,
}

impl<'a, ProcessT: Process> Simulator<'a, ProcessT> {
    pub fn new(config: &'a SimulatorConfig, network: &'a Network, process: &'a ProcessT) -> Self {
        let mut rng = SmallRng::seed_from_u64(0b1110110001110101011000111101); // Doc says that SmallRng should enough 1 in seed
        let state = process.make_initial_state(&mut rng, network);
        assert_eq!(state.node_count(), network.node_count());

        let trace_file = config
            .trace_path
            .as_ref()
            .map(|path| BufWriter::new(File::create(path).unwrap()));

        Simulator {
            network,
            process,
            rng,
            state,
            avg_policies: vec![],
            converged: false,
            step: 0,
            config,
            trace_file,
        }
    }

    fn write_state_trace(&mut self) {
        if let Some(file) = &mut self.trace_file {
            if self.config.report_state_step > 0 && self.step % self.config.report_state_step == 0 {
                let frame = TraceFrame::<'_, ProcessT>::State(StateTraceFrame {
                    step: self.step,
                    states: self.state.node_states(),
                });
                writeln!(file, "{}", serde_json::to_string(&frame).unwrap()).unwrap()
            }
        }
    }

    fn write_window_trace(&mut self, monitor: &Monitor) {
        if let Some(file) = &mut self.trace_file {
            let frame = TraceFrame::<'_, ProcessT>::Window(WindowTraceFrame {
                step: self.step,
                actions: monitor.counts().ncols(),
                counts: monitor.counts().as_slice().unwrap(),
            });
            writeln!(file, "{}", serde_json::to_string(&frame).unwrap()).unwrap()
        }
    }

    pub(crate) fn step(&mut self, stats: &mut Monitor, cache: &mut ProcessT::CacheT) {
        stats.new_step();
        self.step += 1;
        let graph = self.network.graph();
        let node_states = self.state.node_states();
        let mut idx = 0u32;
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
        self.state = State::new(new_node_states);
        self.write_state_trace();
    }

    pub(crate) fn new_monitor(&self) -> Monitor {
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
        self.write_state_trace();
        let mut monitor = self.new_monitor();
        let mut cache = self.process.init_cache();
        for _i in 0..self.config.bootstrap_steps {
            self.step(&mut monitor, &mut cache);
        }
        self.write_window_trace(&monitor);
        drop(monitor);
        for _j in 0..self.config.max_windows {
            let mut monitor = self.new_monitor();
            for _i in 0..self.config.window_steps {
                self.step(&mut monitor, &mut cache);
            }
            self.write_window_trace(&monitor);
            self.avg_policies.push(monitor.avg_policy());
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
