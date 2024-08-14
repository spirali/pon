use rand::distributions::Bernoulli;
use rand::Rng;
use crate::fixarray::{FloatArray, IntArray};
use crate::simulation::{Action, Process};


#[derive(Clone, Default)]
pub(crate) struct MatrixGamePlayerState<const ACTIONS: usize> {
    counts: IntArray<ACTIONS>,
}


pub(crate) struct MatrixGame<const ACTIONS: usize> {
    payoff_matrix: [[f32; ACTIONS]; ACTIONS],
    //error_distribution: Option<Bernoulli>,
}

impl<const ACTIONS:usize> MatrixGame<ACTIONS> {

    pub fn new(payoff_matrix: [[f32; ACTIONS]; ACTIONS]) -> Self {
        MatrixGame {
            payoff_matrix,
        }
    }

    fn expect_payoffs(&self, opponent_strategy: FloatArray<ACTIONS>) -> FloatArray<ACTIONS> {
        FloatArray::from(
            self.payoff_matrix
                .map(|row| opponent_strategy.dot_product(&FloatArray::from(row))),
        )
    }
}

impl<const ACTIONS: usize> Process for MatrixGame<ACTIONS> {
    type State = MatrixGamePlayerState<ACTIONS>;

    fn step(&self, rng: &mut impl Rng, mut state: Self::State, neighbours: impl Iterator<Item=Action>) -> (Self::State, Action) {
        for action in neighbours {
            *state.counts.get_mut(action as usize) += 1;
        }
        /*let action = if rng.sample(self.error_distribution) {
            rng.gen_range(0..ACTIONS)
        } else */
        let action = {
            let policy = state.counts.as_float().normalize_to_policy();
            self.expect_payoffs(policy).argmax()
        };
        (state, action as Action)
    }
}

#[cfg(test)]
mod tests {
    use rand::distributions::Bernoulli;
    use rand::prelude::SmallRng;
    use rand::SeedableRng;
    use crate::matrixgame::{MatrixGame, MatrixGamePlayerState};
    use crate::simulation::Process;

    #[test]
    fn test_game_action() {
        let mut rng = SmallRng::seed_from_u64(0b101010001100011101010110001111);
        let payoffs = [[4.0, 1.0], [3.0, 2.0]];
        let game = MatrixGame {
            payoff_matrix: payoffs,
            //error_distribution: Bernoulli::from_ratio(1, 10).unwrap(),
        };
        let state = MatrixGamePlayerState {
            counts: Default::default(),
        };
        let (new_state, action) = game.step(&mut rng, state, [0].into_iter());
        assert_eq!(new_state.counts.get(0), 1);
        assert_eq!(new_state.counts.get(1), 0);

        let mut state = new_state;
        let mut counts = [0, 0];
        for x in 0..100 {
            let (new_state, action) = game.step(&mut rng, state, [0].into_iter());
            state = new_state;
            counts[action as usize] += 1;
        }
        println!("{:?}", counts);
        assert!(counts[0] > 85);

        let mut state = MatrixGamePlayerState {
            counts: Default::default(),
        };
        let mut counts = [0, 0];
        for x in 0..100 {
            let (new_state, action) = game.step(&mut rng, state, [1].into_iter());
            state = new_state;
            counts[action as usize] += 1;
        }
        println!("{:?}", counts);
        assert!(counts[1] > 85);
    }

}