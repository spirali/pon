use rand::Rng;
use serde::{Serialize, Serializer};
use std::cmp::max;
use std::fmt::{Display, Formatter};
use std::ops::Add;

#[derive(Debug, Clone)]
pub struct FixArray<const SIZE: usize>([f32; SIZE]);

impl<const SIZE: usize> Default for FixArray<SIZE> {
    fn default() -> Self {
        FixArray([0.0f32; SIZE])
    }
}

impl<const SIZE: usize> FixArray<SIZE> {
    #[inline]
    pub fn from(values: [f32; SIZE]) -> Self {
        FixArray(values)
    }

    pub fn sub_scalar(&self, value: f32) -> FixArray<SIZE> {
        let mut result = self.0.clone();
        result.iter_mut().for_each(|x| *x -= value);
        FixArray::from(result)
    }

    #[inline]
    pub fn get(&self, index: usize) -> f32 {
        self.0[index]
    }

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

    pub fn normalize(&self) -> Self {
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
            if value < self.0[i] {
                return i;
            }
            value -= self.0[i];
        }
        return SIZE - 1;
    }
}

impl<const SIZE: usize> Display for FixArray<SIZE> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl<const SIZE: usize> Serialize for FixArray<SIZE> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use crate::base::fixarray::FixArray;
    use approx::assert_abs_diff_eq;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    impl<const SIZE: usize> FixArray<SIZE> {
        fn assert_approx_eq(&self, weights: [f32; SIZE]) {
            self.0.iter().zip(weights).for_each(|(x, y)| {
                assert_abs_diff_eq!(*x, y, epsilon = 0.01);
            })
        }
    }

    fn sample_check<const SIZE: usize>(weights: [f32; SIZE]) {
        let mut counts = [0.0; SIZE];
        let mut rng = SmallRng::seed_from_u64(0b101010001100011101010110001111); // Doc says that SmallRng should enough 1 in seed
        let f = FixArray::from(weights);
        const COUNT: usize = 10000;
        for _i in 0..COUNT {
            counts[f.sample_index(&mut rng)] += 1.0;
        }
        counts.iter_mut().for_each(|x| *x /= COUNT as f32);
        dbg!(&counts);

        let n = f.normalize();
        n.assert_approx_eq(counts);
    }

    #[test]
    fn test_fixarray_sample() {
        sample_check([100.0, 100.0, 100.0]);
        sample_check([10.0, 0.0, 1.0]);
        sample_check([1.0, 2.0, 3.0]);
    }
}
