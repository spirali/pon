use crate::process::network::Network;
use crate::process::process::Process;
use crate::process::report::RunReport;
use crate::process::state::State;
use crate::process::utils::max_of_array;
use ndarray::{Array2, Axis};
use rand::rngs::{SmallRng, ThreadRng};
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
    counts: &'a [u64],
}

impl SimulatorConfig {
    pub fn new() -> Self {
        SimulatorConfig {
            bootstrap_steps: 5000,
            window_steps: 200,
            max_windows: 1000,
            termination_threshold: 0.001,
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
    //rng: ThreadRng,
    state: State<ProcessT>,

    action_counts: Array2<u64>,
    last_policies: Array2<f32>,
    converged: bool,
    step: usize,

    config: &'a SimulatorConfig,
    trace_file: Option<BufWriter<File>>,
}

impl<'a, ProcessT: Process> Simulator<'a, ProcessT> {
    pub fn new(
        config: &'a SimulatorConfig,
        outer_rng: Option<&mut ThreadRng>,
        network: &'a Network,
        process: &'a ProcessT,
    ) -> Self {
        let mut rng = outer_rng
            .map(|r| SmallRng::from_rng(r).unwrap())
            .unwrap_or_else(|| {
                SmallRng::seed_from_u64(0b1110110001110101011000111101) // Doc says that SmallRng should enough 1 in seed
                                                                        //let mut rng = rand::thread_rng();
            });
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
            action_counts: Array2::zeros((network.node_count(), ProcessT::ACTIONS)),
            last_policies: Array2::zeros((network.node_count(), ProcessT::ACTIONS)),
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

    fn write_window_trace(&mut self) {
        if let Some(file) = &mut self.trace_file {
            let frame = TraceFrame::<'_, ProcessT>::Window(WindowTraceFrame {
                step: self.step,
                actions: self.action_counts.ncols(),
                counts: self.action_counts.as_slice().unwrap(),
            });
            writeln!(file, "{}", serde_json::to_string(&frame).unwrap()).unwrap()
        }
    }

    pub(crate) fn step(&mut self) {
        self.step += 1;
        let graph = self.network.graph();
        let node_states = self.state.node_states();
        let last_actions = self.state.last_actions();
        let mut idx = 0u32;
        let (new_node_states, new_actions): (Vec<_>, Vec<_>) = node_states
            .iter()
            .zip(last_actions)
            .map(|(node_state, last_action)| {
                let neighbors = graph
                    .neighbors(idx.into())
                    .map(|other| last_actions[other.index()]);
                let (new_state, action) =
                    self.process
                        .node_step(&mut self.rng, node_state, *last_action, neighbors);
                self.action_counts[(idx as usize, action)] += 1;
                idx += 1;
                (new_state, action)
            })
            .unzip();
        self.state = State::new(new_node_states, new_actions);
        self.write_state_trace();
    }

    pub fn report(&self) -> RunReport {
        RunReport {
            steps: self.step,
            converged: self.converged,
            avg_policy: self.last_policies.clone(),
        }
    }

    pub fn reset_counts(&mut self) {
        self.action_counts.fill(0);
    }

    pub fn compute_policies(&self) -> Array2<f32> {
        let sums = self.action_counts.sum_axis(Axis(1)).mapv(|x| x as f32);
        self.action_counts.mapv(|x| x as f32) / sums.insert_axis(Axis(1))
    }

    pub fn run(&mut self) -> bool {
        self.write_state_trace();
        for _i in 0..self.config.bootstrap_steps {
            self.step();
        }
        self.write_window_trace();
        self.reset_counts();
        for _j in 0..self.config.max_windows {
            for _i in 0..self.config.window_steps {
                self.step();
            }
            self.write_window_trace();
            if self.check_termination() {
                self.converged = true;
                return true;
            }
        }
        false
    }

    fn check_termination(&mut self) -> bool {
        let policies = self.compute_policies();
        let v = max_of_array(&policies - &self.last_policies);
        self.last_policies = policies;
        v < self.config.termination_threshold
    }

    pub fn state(&mut self) -> &State<ProcessT> {
        &self.state
    }
}
