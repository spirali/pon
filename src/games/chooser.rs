use crate::games::game::ActionId;
use crate::process::fixarray::FloatArray;
use rand::distributions::Bernoulli;
use rand::Rng;

pub trait ActionChooser<const ACTIONS: usize> {
    fn choose_action(&self, rng: &mut impl Rng, payoffs: FloatArray<ACTIONS>) -> ActionId;
}

pub struct DirectChooser;

impl DirectChooser {
    pub fn new() -> Self {
        DirectChooser {}
    }
}

impl<const ACTIONS: usize> ActionChooser<ACTIONS> for DirectChooser {
    fn choose_action(&self, rng: &mut impl Rng, payoffs: FloatArray<ACTIONS>) -> ActionId {
        payoffs.sample_index(rng)
    }
}

pub struct SamplingEpsilonError {
    error_distribution: Bernoulli,
}

impl SamplingEpsilonError {
    pub fn new(epsilon: f32) -> Self {
        SamplingEpsilonError {
            error_distribution: Bernoulli::new(epsilon as f64).unwrap(),
        }
    }
}

impl<const ACTIONS: usize> ActionChooser<ACTIONS> for SamplingEpsilonError {
    fn choose_action(&self, rng: &mut impl Rng, payoffs: FloatArray<ACTIONS>) -> ActionId {
        if rng.sample(&self.error_distribution) {
            rng.gen_range(0..ACTIONS)
        } else {
            payoffs.sample_index(rng)
        }
    }
}

pub struct BestResponseEpsilonError {
    error_distribution: Bernoulli,
}

impl BestResponseEpsilonError {
    pub fn new(epsilon: f32) -> Self {
        BestResponseEpsilonError {
            error_distribution: Bernoulli::new(epsilon as f64).unwrap(),
        }
    }
}

impl<const ACTIONS: usize> ActionChooser<ACTIONS> for BestResponseEpsilonError {
    fn choose_action(&self, rng: &mut impl Rng, payoffs: FloatArray<ACTIONS>) -> ActionId {
        if rng.sample(&self.error_distribution) {
            rng.gen_range(0..ACTIONS)
        } else {
            payoffs.argmax()
        }
    }
}

#[derive(Default)]
pub struct SoftmaxSample;

impl<const ACTIONS: usize> ActionChooser<ACTIONS> for SoftmaxSample {
    fn choose_action(&self, rng: &mut impl Rng, payoffs: FloatArray<ACTIONS>) -> ActionId {
        payoffs.exp().sample_index(rng)
    }
}
