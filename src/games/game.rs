use crate::process::fixarray::FixArray;

pub type ActionId = usize;

#[derive(Debug, Clone)]
pub enum InitialAction<const ACTIONS: usize> {
    Const(ActionId),
    Uniform,
    Distribution(FixArray<ACTIONS>),
}

#[derive(Debug, Clone)]
pub struct MatrixGame<const ACTIONS: usize> {
    payoff_matrix: [[f32; ACTIONS]; ACTIONS],
    initial_action: InitialAction<ACTIONS>,
}

impl<const ACTIONS: usize> MatrixGame<ACTIONS> {
    pub fn new(
        payoff_matrix: [[f32; ACTIONS]; ACTIONS],
        initial_action: InitialAction<ACTIONS>,
    ) -> Self {
        assert!(ACTIONS > 0);
        assert!(match initial_action {
            InitialAction::Const(a) => a <= ACTIONS,
            InitialAction::Uniform | InitialAction::Distribution(_) => true,
        });
        MatrixGame {
            payoff_matrix,
            initial_action,
        }
    }

    pub fn make_initial_action(&self, rng: &mut impl rand::Rng) -> ActionId {
        match &self.initial_action {
            InitialAction::Const(action) => *action,
            InitialAction::Uniform => rng.gen_range(0..ACTIONS),
            InitialAction::Distribution(a) => a.sample_index(rng),
        }
    }

    #[inline(always)]
    fn update_payoffs(&self, payoffs: &mut [f32; ACTIONS], opponent_action: ActionId) {
        for (i, p) in payoffs.iter_mut().enumerate() {
            *p += self.payoff_matrix[i][opponent_action];
        }
    }

    pub fn payoffs_sums(&self, actions: impl Iterator<Item = ActionId>) -> FixArray<ACTIONS> {
        let mut payoffs = [0f32; ACTIONS];
        actions.for_each(|a| self.update_payoffs(&mut payoffs, a));
        FixArray::from(payoffs)
    }
}
