use ndarray::{s, Axis, Array, Array1, Array2, concatenate};
use rand_distr::{Normal, Distribution};
use rand::{self};
use crate::Prec;

/// Create spiral data set.
/// outputs `(x, y)` where `x` is the data matrix and `y` is the label vector.
/// See <https://cs231n.github.io/neural-networks-case-study/>
pub fn spiral_data(samples_per_class: usize, n_classes: usize) -> (Array2<Prec>, Array1<usize>){

    let n = samples_per_class; // number of points per class
    let d = 2; // dimensionality
    let k = n_classes; // number of classes

    let mut x = Array2::zeros((n*k, d)); //# data matrix (each row = single example)
    let mut y = Array1::<usize>::zeros(n*k); //# class labels
    for j in 0.. k {

        let ix = n*j..n*(j+1);

        let r = Array::linspace(0.0, 1.0, n); // radius
        let t = Array::linspace((j*4) as Prec,((j+1)*4) as Prec, n) + random_vector(n)*0.2; // theta

        let rt = concatenate![
            Axis(1),
            (t.mapv(f32::sin) * &r).insert_axis(Axis(1)).to_owned(),
            (t.mapv(f32::cos) * &r).insert_axis(Axis(1)).to_owned()
        ];

                
        x.slice_mut(s![ix.clone(), ..]).assign(&rt);
        y.slice_mut(s![ix]).fill(j);
    }   
    
    (x, y)
}