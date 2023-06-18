use nnfs::{*, layer::*, activation::*};
use ndarray::{Array1, Array2, Axis};
use vec_box::vec_box;

fn main() {
    let (x,y_true) = dataset::spiral_data(10, 3);

    plot::plot_2d("spiral_actual.png", &x, &y_true).unwrap();

    let x2 = x.clone();

    let network: Vec<Box<dyn Layer>> = vec_box![
        LayerDense::new(2,3),
        ActivationFn::Relu,
        LayerDense::new(3,3),
        ActivationFn::Softmax,
        ];

    let forward_pass = |network: Vec<Box<dyn Layer>>, input| {network.iter().fold(input, |step, layer| layer.forward(step))};

    let y_prob = forward_pass(network, x);
    let y_pred = util::argmax(&y_prob);

    println!("{:#?}", y_prob);
    println!("{:#?}", y_pred);

    plot::plot_2d("spiral_class.png", &x2, &y_pred).unwrap();

    let loss = categorical_loss_entropy(&y_prob, &y_true);
    println!("loss: {:#?}", loss);
    println!("y_len: {:}", y_true.len());
}



pub fn categorical_loss_entropy(y_prob: &Array2<Prec>, y_true: &Array1<usize>) -> Prec {
    // Iterate over each row of probabilities and the corresponding true class labels
     let sum_of_loss: Prec = y_prob.axis_iter(Axis(0)) 
    .zip(y_true)
    // Extract the probability predicted for the true class
    .map( |(row, class)| row[*class]) 
    // Clip the probability to avoid undefined results from taking the logarithm of 0 or 1
    .map( |prob| prob.max(1e-7).min(1.0 - 1e-7))
    // Compute loss for this classification
    .map( |prob| -prob.ln()) 
    // Sum up all the losses
    .sum();

    // Return the average loss
    sum_of_loss / y_true.len() as Prec
} 