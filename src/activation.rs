use ndarray::{Array1, Array2};
use crate::{Prec};


pub enum ActivationFn {
    Relu,
    Linear,
    Sigmoid,
    Softmax,
}

impl ActivationFn {
    pub fn apply(&self, input: Array2<Prec>) -> Array2<Prec> {
        match self {
            ActivationFn::Relu => relu(input),
            ActivationFn::Linear => linear(input),
            ActivationFn::Sigmoid => sigmoid(input),
            ActivationFn::Softmax => softmax(input),
        }
    }
}


pub struct LayerActivation {
    func: ActivationFn,
}

impl LayerActivation {
    pub fn new(func: ActivationFn) -> LayerActivation{
        LayerActivation { func }
    }
}

impl Layer for LayerActivation {
    fn forward(&self, input: Array2<Prec> ) -> Array2<Prec> {
        self.func.apply(input)
     }
}



pub fn relu(inputs: Array2<Prec>) -> Array2<Prec> {
    inputs.mapv(|x| x.max(0.0))
}

pub fn linear(inputs: Array2<Prec>) -> Array2<Prec> {
    inputs
}

pub fn sigmoid(inputs: Array2<Prec>) -> Array2<Prec> {
    inputs.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}


pub fn softmax(inputs: Array2<Prec>) -> Array2<Prec> {
    let max_row = inputs.fold_axis(Axis(0), Prec::MIN, |&a, &b| a.max(b));
    let exp = (&inputs - &max_row).mapv(|x| x.exp());
    let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
    exp / sum
}