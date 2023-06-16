use ndarray::{Array1, Array2, };

pub mod layer;
pub mod data;
//pub mod activation;

pub type Prec = f32;

pub type ActivationFn = fn(Array2<Prec>) -> Array2<Prec>;
