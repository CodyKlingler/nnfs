

/// Computes the row-wise argmax of a matrix.
/// Used to turn matrix of classification probabilities or one-hot encoding into vector of class values.
pub fn argmax(y_prob: &Array2<Prec>) -> Array1<usize> {
    y_prob.map_axis(Axis(1), |row| row.into_iter() // reduce each row to a single element
    .enumerate()    // iterate over values and index
    .fold( (0 as usize, Prec::MIN), |old, new| if old.1 > *new.1 {old} else {(new.0, *new.1)})) // retain the tuple with the max value
    .map(|x| x.0) //retain indices from tuple
}

/// Univariate “normal” (Gaussian) distribution of mean `0` and variance `1`
pub fn random_matrix(n:usize, m:usize) -> Array2<Prec> {
    let normal_distribution = Normal::new(0., 1.).unwrap();
    let rand_norm = || normal_distribution.sample(&mut rand::thread_rng());
    Array::from_shape_simple_fn((n, m), rand_norm)
}

/// Univariate “normal” (Gaussian) distribution of mean `0` and variance `1`
pub fn random_vector(n:usize) -> Array1<Prec> {
    let normal_distribution = Normal::new(0., 1.).unwrap();
    let rand_norm = || normal_distribution.sample(&mut rand::thread_rng());
    Array1::from_shape_simple_fn(n, rand_norm)
}
