use nnfs::{layer::*, activation::*, *};
use ndarray::{Array1, Array2, Axis};

fn main() {
    let (x,y) = nnfs::data::spiral_data(600, 3);

    plot::plot_2d("spiral_actual.png", &x, &y).unwrap();

    let x2 = x.clone();

    let l1 = LayerDense::new(2,3);
    let a1 = LayerActivation::new(ActivationFn::Relu);
    let l2 = LayerDense::new(3,3);
    let a2 = LayerActivation::new(ActivationFn::Softmax);
    
    let mut step;
    step = l1.forward(x);
    step = a1.forward(step);
    step = l2.forward(step);
    step = a2.forward(step);

    println!("{:}", step);

    let predicted_classes = argmax(&step);


    println!("{:#?}", step);

    plot::plot_2d("spiral_class.png", &x2, &predicted_classes).unwrap();

    let loss = categorical_loss_entropy(&step, &y);
    println!("loss: {:#?}", loss);
    println!("y_len: {:}", y.len());
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