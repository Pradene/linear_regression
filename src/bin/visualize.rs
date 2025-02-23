use linear_regression::{load_data, load_thetas};
use plotters::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    
    let (x, y) = load_data()?;
    let thetas = load_thetas()?;
    
    let root = BitMapBackend::new("data.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_range = 0.0 .. 300_000.0;
    let y_range = 0.0 .. 10_000.0;

    let mut chart = ChartBuilder::on(&root)
        .caption("Car Price based on km", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_range.clone(), y_range.clone())?;

    chart
        .configure_mesh()
        .x_desc("Km")
        .y_desc("Price")
        .draw()?;

    let points: Vec<_> = x.iter().zip(y.iter()).map(|(x, y)| (x.clone(), y.clone())).collect();
    chart.draw_series(
        points.iter()
            .map(|(x, y)| {
                Circle::new((*x, *y), 5, ShapeStyle::from(&BLUE).filled())
            })
    )?;

    let linear_function = |x: f64| thetas[0] + thetas[1] * x;
    chart.draw_series(LineSeries::new(
        (0..100).map(|i| {
            let x = x_range.start + (x_range.end - x_range.start) * (i as f64) / 100.0;
            (x, linear_function(x))
        }),
        RED.stroke_width(2),
    ))?;

    Ok(())
}