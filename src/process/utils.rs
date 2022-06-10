use ndarray::Array2;

pub(crate) fn max_of_array(array: Array2<f32>) -> f32 {
    array.iter().fold(
        0.0,
        |val_max, val| {
            if &val_max > val {
                val_max
            } else {
                *val
            }
        },
    )
}
