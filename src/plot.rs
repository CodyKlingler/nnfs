
use plotters::prelude::*;
use plotters;
use crate::{*};
use ndarray::{Array1, Array2, Axis};


// TODO: add more colors. make view window dynamic.
pub fn plot_2d(plot_name: &str, x: &Array2<f32>, y: &Array1<usize>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(plot_name, (640*2, 480*2)).into_drawing_area();
    root.fill(&WHITE)?;

    let mins = x.fold_axis(Axis(0), Prec::MAX, |&a, &b| a.min(b));
    let maxs = x.fold_axis(Axis(0), Prec::MIN, |&a, &b| a.max(b));
    let min_x = *mins.get(0).unwrap();
    let max_x = *maxs.get(0).unwrap();
    let min_y = *mins.get(1).unwrap();
    let max_y = *maxs.get(1).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Scatter Plot", ("Arial", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_x.. max_x, min_y.. max_y)?;

    chart.configure_mesh().draw()?;

    for (index, element) in y.iter().enumerate() {
        let color = match *element {
            0 => RED,
            1 => GREEN,
            2 => BLUE,
            3 => MAGENTA,
            4 => CYAN,
            5 => YELLOW,
            6 => RGBColor(127, 127, 127),
            _ => BLACK,
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


