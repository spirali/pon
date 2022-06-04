use rand::Rng;
use std::cmp::max;
use std::ops::Add;

#[derive(Debug, Clone)]
pub struct FixArray<const SIZE: usize>([f32; SIZE]);

impl<const SIZE: usize> FixArray<SIZE> {
    #[inline]
    pub fn from(values: [f32; SIZE]) -> Self {
        FixArray(values)
    }
}

impl<const SIZE: usize> Default for FixArray<SIZE> {
    fn default() -> Self {
        FixArray([0.0f32; SIZE])
    }
}

impl<const SIZE: usize> FixArray<SIZE> {
    pub fn add(&self, other: &Self) -> Self {
        let mut result = [0.0; SIZE];
        for i in 0..SIZE {
            result[i] = self.0[i] + other.0[i];
        }
        FixArray::from(result)
    }

    pub fn sum(&self) -> f32 {
        self.0.iter().sum()
    }

    pub fn normalize(&mut self) -> Self {
        debug_assert!(self.is_finite());
        let sum = self.sum();
        if sum <= 0.0 {
            return self.clone();
        }
        let mut result = self.0;
        for i in 0..SIZE {
            result[i] /= sum;
        }
        FixArray::from(result)
    }

    pub fn clamp_negatives(&self) -> Self {
        let mut result = [0.0; SIZE];
        for i in 0..SIZE {
            result[i] = 0.0f32.max(self.0[i]);
        }
        FixArray::from(result)
    }

    pub fn argmax(&self) -> usize {
        let (max_idx, _) =
            self.0
                .iter()
                .enumerate()
                .fold((0, self.0[0]), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                });
        max_idx
    }

    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }

    pub fn sample_index(&self, rng: &mut impl Rng) -> usize {
        debug_assert!(self.is_finite());
        let sum = self.sum();
        if sum <= 0.0 {
            return rng.gen_range(0..SIZE);
        }
        let mut value: f32 = rng.gen_range(0.0..sum);
        for i in 0..SIZE - 1 {
            if self.0[i] < value {
                return i;
            }
        }
        return SIZE - 1;
    }
}
