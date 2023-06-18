use ndarray::{Array1, Array2};
use crate::{*};

pub struct LayerDense  {
    pub weights: Array2<Prec>,
    pub biases:  Array1<Prec>,
}


pub trait Layer {
    fn forward(&self, input: Array2<Prec> ) -> Array2<Prec>;
}

impl LayerDense {

    pub fn new(n_inputs: usize, n_neurons: usize) -> LayerDense{

        let weights = util::random_matrix(n_inputs, n_neurons);
        let biases = Array1::zeros(n_neurons);

        LayerDense { weights, biases }
    }

}

impl Layer for LayerDense {
    fn forward(&self, input: Array2<Prec> ) -> Array2<Prec> {
       input.dot(&self.weights) + &self.biases
    }
}


