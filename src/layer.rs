use ndarray::{Array1, Array2, Axis, Array};
use crate::{Prec, ActivationFn};

pub struct LayerDense  {
    pub weights: Array2<Prec>,
    pub biases:  Array1<Prec>,
}


pub trait Layer {
    fn forward(&self, input: Array2<Prec> ) -> Array2<Prec>;
}

impl LayerDense {

    pub fn new(n_inputs: usize, n_neurons: usize) -> LayerDense{

        let weights = crate::data::random_matrix(n_inputs, n_neurons);
        let biases = Array1::zeros(n_neurons);

        LayerDense { weights, biases }
    }

}

impl Layer for LayerDense {
    fn forward(&self, input: Array2<Prec> ) -> Array2<Prec> {
       input.dot(&self.weights) + self.biases.clone()
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
        (self.func)(input)
     }
}


pub fn relu(mut inputs: Array2<Prec>) -> Array2<Prec> {
    inputs.mapv_inplace(|x| Prec::max(0.0,x));
    inputs
}

pub fn linear(inputs: Array2<Prec>) -> Array2<Prec> {
    inputs
}

pub fn sigmoid(mut inputs: Array2<Prec>) -> Array2<Prec> {

    let sig = |x: Prec| 1.0 / (1.0 + 1.0/x.exp());
    inputs.mapv_inplace(sig);
    inputs
}

pub fn softmax(inputs: Array2<Prec>) -> Array2<Prec> {
    let max_row = inputs.map_axis(Axis(0), |x| *x.iter().max_by(|a, b| a.total_cmp(b)).unwrap());
    let neg_input = inputs.clone() - max_row;
    let exp = neg_input.mapv(|x| x.exp()); // mapv_inplace neg_input instead?
    let sum_row = exp.sum_axis(Axis(1));
    let softmax = exp / sum_row;
    
    softmax
}