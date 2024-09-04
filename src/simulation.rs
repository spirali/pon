use std::cmp::max;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::Arc;
use petgraph::{Graph, Undirected};
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

pub(crate) type Action = u32;
pub(crate) type Step = u32;
pub(crate) type SimGraph = Graph<(), (), Undirected>;

pub(crate) trait Process {
    type State: Default;

    fn step(&self, rng: &mut impl Rng, state: Self::State, neighbours: impl Iterator<Item=Action>) -> (Self::State, Action);
}

#[derive(Debug)]
pub(crate) struct Simulation<'a, P: Process> {
    graph: &'a SimGraph,
    process: &'a P,
    state: Vec<P::State>,
    actions: Vec<Action>,
}

pub(crate) enum EndReason {
    Converged,
    MeanCheck,
    MaxStepsReached,
}

pub(crate) struct RunResult {
    pub history: Vec<(u32, Vec<Action>)>,
    pub means: Vec<f32>,
    pub end_reason: EndReason,
}

pub(crate) struct RunConfig {
    pub max_steps: usize,
    pub store_steps: usize,
}

impl<'a, P: Process> Simulation<'a, P> {
    pub fn new<F: FnMut() -> Action>(graph: &'a SimGraph, process: &'a P, mut init_action: F) -> Self {
        let state = (0..graph.node_count()).map(|_| P::State::default()).collect();
        let actions = (0..graph.node_count()).map(|_| init_action()).collect();
        Simulation {
            graph,
            process,
            state,
            actions,
        }
    }

    pub fn step(&mut self, rng: &mut SmallRng) -> Vec<Action> {
        let state = std::mem::take(&mut self.state);
        let mut new_states = Vec::with_capacity(state.len());
        let mut new_actions = Vec::with_capacity(state.len());
        let mut idx = 0;
        for s in state {
            let neighbors = self.graph
                .neighbors(idx.into())
                .map(|other| self.actions[other.index()]);
            let (new_state, new_action) = self.process.step(rng, s, neighbors);
            new_states.push(new_state);
            new_actions.push(new_action);
            idx += 1;
        }
        self.state = new_states;
        std::mem::replace(&mut self.actions, new_actions)
    }

    pub fn run(&mut self, rng: impl Rng, config: &RunConfig) -> RunResult {
        const LAST_CHECK: usize = 10;
        const BOOSTRAP: usize = 500;
        const MEAN_CHECK_WINDOW: usize = 64;
        const MEAN_CHECK_THRESHOLD: f32 = 0.0001;

        let mut small_rng = SmallRng::from_rng(rng).unwrap();
        let mut history = Vec::new();
        let mut means = Vec::new();
        let mut step = 0;
        let mut last_counter = 0;
        let mut end_reason = EndReason::MaxStepsReached;

        for i in 0..config.max_steps {
            if (step % config.store_steps == 0) {
                history.push((step as u32, self.actions.clone()))
            }
            step += 1;
            let last = self.step(&mut small_rng);

            means.push(last.iter().sum::<Action>() as f32 / last.len() as f32);

            if last == self.actions {
                last_counter += 1;
                if last_counter >= LAST_CHECK {
                    end_reason = EndReason::Converged;
                    break;
                }
            } else {
                last_counter = 0;
            }

            if i > BOOSTRAP {
                let window1 = &means[means.len() - 2 * MEAN_CHECK_WINDOW..];
                let window2 = &means[means.len() - MEAN_CHECK_WINDOW..];
                if (window1.iter().sum::<f32>() / window1.len() as f32) - (window2.iter().sum::<f32>() / window2.len() as f32).abs() < MEAN_CHECK_THRESHOLD {
                    end_reason = EndReason::MeanCheck;
                    break
                }
            }
        }
        history.push((step as u32, self.actions.clone()));
        RunResult {
            history,
            means,
            end_reason,
        }
    }
}
