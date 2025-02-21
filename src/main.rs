use std::error::Error;
use std::fs::File;
use serde::Deserialize;

const STEPS: u32 = 500;
const RATE: f64 = 0.01;

#[derive(Debug, Deserialize)]
struct Record {
    km: u32,
    price: u32
}

fn load_data(path: &str) -> Result<(ndarray::Array1<f64>, ndarray::Array1<f64>), Box<dyn std::error::Error>> {
    let mut features = Vec::new();
    let mut targets = Vec::new();

    let file = File::open(path)?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    for result in reader.deserialize() {
        let record: Record = result?;
        targets.push(record.price as f64);
        features.push(record.km as f64);
    }

    let x = ndarray::Array1::from(features);
    let y = ndarray::Array1::from(targets);

    Ok((x, y))
}

fn normalize(data: &mut ndarray::Array1<f64>) -> (f64, f64) {
    let mean = data.mean().unwrap();
    let std = data.std(0.0);
    
    data.mapv_inplace(|v| (v - mean) / std);

    (mean, std)
}

fn reshape(data: &ndarray::Array1<f64>) -> ndarray::Array2<f64> {
    let n = data.len();
    let ones = ndarray::Array2::ones((n, 1));
    let x = data.view().insert_axis(ndarray::Axis(1));
    
    ndarray::concatenate(ndarray::Axis(1), &[ones.view(), x.view()]).unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    
    let path = "data.csv";
    let (mut x, mut y) = load_data(path)?;
    let mut thetas = ndarray::Array1::zeros(2);

    let (mean_x, std_x) = normalize(&mut x);
    let (mean_y, std_y) = normalize(&mut y);
    
    let x = reshape(&x);

    for _ in 0 .. STEPS {
        let predictions = x.dot(&thetas);
        let errors = &predictions - &y;
        let gradients = 1. / x.nrows() as f64 * x.t().dot(&errors);
        thetas = thetas - RATE * gradients;
    }

    let theta1 = thetas[1] * (std_y / std_x);
    let theta0 =  (thetas[0] * std_y) + mean_y - theta1 * mean_x;
    
    println!("{}", theta0);
    println!("{}", theta1);
    
    Ok(())
}
