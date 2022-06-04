use crate::base::fixarray::FixArray;

pub type ActionId = usize;

#[derive(Debug, Clone)]
pub enum InitialAction {
    Const(ActionId),
    Random,
}

#[derive(Debug, Clone)]
pub struct MatrixGame<const NACTIONS: usize> {
    payoff_matrix: [[f32; NACTIONS]; NACTIONS],
    initial_action: InitialAction,
}

impl<const NACTIONS: usize> MatrixGame<NACTIONS> {
    pub fn new(payoff_matrix: [[f32; NACTIONS]; NACTIONS], initial_action: InitialAction) -> Self {
        assert!(NACTIONS > 0);
        assert!(match initial_action {
            InitialAction::Const(a) => a <= NACTIONS,
            InitialAction::Random => true,
        });
        MatrixGame {
            payoff_matrix,
            initial_action,
        }
    }

    pub fn make_initial_action(&self, rng: &mut impl rand::Rng) -> ActionId {
        match self.initial_action {
            InitialAction::Const(action) => action,
            InitialAction::Random => rng.gen_range(0..NACTIONS),
        }
    }

    #[inline(always)]
    fn update_payoffs(&self, payoffs: &mut [f32; NACTIONS], opponent_action: ActionId) {
        for (i, p) in payoffs.iter_mut().enumerate() {
            *p += self.payoff_matrix[i][opponent_action];
        }
    }

    pub fn payoffs_sums(&self, actions: impl Iterator<Item = ActionId>) -> FixArray<NACTIONS> {
        let mut payoffs = [0f32; NACTIONS];
        actions.for_each(|a| self.update_payoffs(&mut payoffs, a));
        FixArray::from(payoffs)
    }
}
