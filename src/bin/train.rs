use std::error::Error;
use ndarray::{Array1, Array2, Axis};

use linear_regression::{load_data, save_thetas};

const STEPS: u32 = 500;
const RATE: f64 = 0.01;

fn normalize(data: &mut Array1<f64>) -> (f64, f64) {
    let mean = data.mean().unwrap();
    let std = data.std(0.0);
    
    data.mapv_inplace(|v| (v - mean) / std);

    (mean, std)
}

fn add_bias(x: &Array1<f64>) -> Array2<f64> {
    let n = x.len();
    let ones = Array2::ones((n, 1));
    let x = x.view().insert_axis(Axis(1));
    
    ndarray::concatenate(Axis(1), &[ones.view(), x.view()]).unwrap()
}

fn cost(x: &Array2<f64>, y: &Array1<f64>, thetas: &Array1<f64>) -> f64 {
    let m = x.len() as f64;
    let p = x.dot(thetas);
    let e = &p - y;
    1. / (2. * m) * e.dot(&e)
}

fn gradient(x: &Array2<f64>, y: &Array1<f64>, thetas: &Array1<f64>) -> Array1<f64> {
    let m = x.nrows() as f64;

    1. / m * x.t().dot(&(x.dot(thetas) - y))
}

fn main() -> Result<(), Box<dyn Error>> {
    let (mut x, mut y) = load_data()?;
    let mut thetas = Array1::zeros(2);

    let (mean_x, std_x) = normalize(&mut x);
    let (mean_y, std_y) = normalize(&mut y);

    let x = add_bias(&x);

    for _ in 0 .. STEPS {
        let gradients = gradient(&x, &y, &thetas);
        thetas = thetas - RATE * gradients;

        let c = cost(&x, &y, &thetas);
        println!("{}", c);
    }

    // Denormalize thetas because trained on normalized values
    thetas[1] = thetas[1] * (std_y / std_x);
    thetas[0] =  (thetas[0] * std_y) + mean_y - thetas[1] * mean_x;

    save_thetas(&thetas)?;

    Ok(())
}
