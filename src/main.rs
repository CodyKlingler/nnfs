use plotters;
use nnfs::{layer::*, Prec};
use ndarray::{Array1, Array2, Axis, s};

fn main() {


    let (x,y) = nnfs::data::spiral_data(100, 3);

    let l1 = LayerDense::new(2,3);
    let a1 = LayerActivation::new(relu);
    let l2 = LayerDense::new(3,3);
    let a2 = LayerActivation::new(softmax);
    
    let mut step;
    step = l1.forward(x);
    step = a1.forward(step);
    step = l2.forward(step);
    step = a2.forward(step);

    println!("{:}", step);

    //plot_data(&x, &y).unwrap();
}

use plotters::prelude::*;


// TODO: add more colors. make view window dynamic.
fn plot_data(x: &Array2<f32>, y: &Array1<u8>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("plot.png", (640*2, 480*2)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Scatter Plot", ("Arial", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.0f32 .. 1.0f32, -1.0f32.. 1.0f32)?;

    chart.configure_mesh().draw()?;

    for (index, element) in y.iter().enumerate() {
        let color = match *element {
            0 => RED,
            1 => GREEN,
            _ => BLUE,
        };

        let point: (f32, f32) = (*x.get((index, 0)).unwrap(), *x.get((index, 1)).unwrap());
        let point = vec![point];
        chart.draw_series(PointSeries::of_element(
            point, 
            5, 
            color,
            &|c, s, st| {
                return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
                + Circle::new((0,0),s,st.filled()) // At this point, the new pixel coordinate is established
                //+ Text::new(format!("{:?}", c), (10, 0), ("sans-serif", 10).into_font());
            }
        ))?;
    }

    root.present()?;
    Ok(())
}




