use std::error::Error;
use ndarray::{Array1, Array2, Axis};
use linear_regression::{load_data, load_thetas, model};

fn add_bias(x: &Array1<f64>) -> Array2<f64> {
    let n = x.len();
    let ones = Array2::ones((n, 1));
    let x = x.view().insert_axis(Axis(1));
    
    ndarray::concatenate(Axis(1), &[ones.view(), x.view()]).unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    let (x, y) = load_data()?;
    let thetas = load_thetas()?;

    let x = add_bias(&x);

    let p = model(&x, &thetas);

    let mut sum = 0.0;
    for (y, p) in y.iter().zip(p.iter()) {
        if *y != 0.0 {
            sum += ((p - y).abs() / y.abs()) * 100.0;
        }
    }

    let mae = sum / y.len() as f64;

    println!("{}", mae);

    Ok(())
}