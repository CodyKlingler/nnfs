use ndarray::{Array2, Axis};
use crate::{Prec, layer::*};


pub enum ActivationFn {
    Relu,
    Linear,
    Sigmoid,
    Softmax,
}

impl Layer for ActivationFn {
    fn forward(&self, input: Array2<Prec>) -> Array2<Prec> {
        match self {
            ActivationFn::Relu => relu(input),
            ActivationFn::Linear => linear(input),
            ActivationFn::Sigmoid => sigmoid(input),
            ActivationFn::Softmax => softmax(input),
        }
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
    let max_row = inputs.fold_axis(Axis(1), Prec::MIN, |&a, &b| a.max(b)).insert_axis(Axis(1));
    let exp = (&inputs - &max_row).mapv(|x| x.exp());
    let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
    exp / sum
}