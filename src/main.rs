use nnfs::{layer::*, activation::*, *};
use ndarray::{Array1, Array2, Axis};

fn main() {
    let (x,y) = nnfs::data::spiral_data(100, 3);

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

    let class = argmax(step);

    plot::plot_2d("spiral_class.png", &x2, &class).unwrap();
}

/// Takes a matrix of classification probabilities or one-hot encoding and returns 
/// a vector of 
pub fn argmax(output: Array2<Prec>) -> Array1<usize> {
     /* nasty way to turn softmax output into class labels */
     let class = output.map_axis(Axis(1), |row| row.into_iter()
     .enumerate()
     .fold( (0 as usize, Prec::MIN), |old, new| if old.1 > *new.1 {old} else {(new.0, *new.1)}));                                

    class.map(|x| x.0)
}
