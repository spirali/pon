use crate::games::game::ActionId;
use crate::process::fixarray::FixArray;
use rand::distributions::Bernoulli;
use rand::Rng;

pub trait ActionChooser<const ACTIONS: usize> {
    fn choose_action(&self, rng: &mut impl Rng, policy: FixArray<ACTIONS>) -> ActionId;
}

pub struct DirectChooser;

impl DirectChooser {
    pub fn new() -> Self {
        DirectChooser {}
    }
}

impl<const ACTIONS: usize> ActionChooser<ACTIONS> for DirectChooser {
    fn choose_action(&self, rng: &mut impl Rng, policy: FixArray<ACTIONS>) -> ActionId {
        policy.sample_index(rng)
    }
}

pub struct EpsilonError {
    error_distribution: Bernoulli,
}

impl EpsilonError {
    pub fn new(epsilon: f32) -> Self {
        EpsilonError {
            error_distribution: Bernoulli::new(epsilon as f64).unwrap(),
        }
    }
}

impl<const ACTIONS: usize> ActionChooser<ACTIONS> for EpsilonError {
    fn choose_action(&self, rng: &mut impl Rng, policy: FixArray<ACTIONS>) -> ActionId {
        if rng.sample(&self.error_distribution) {
            rng.gen_range(0..ACTIONS)
        } else {
            policy.sample_index(rng)
        }
    }
}
