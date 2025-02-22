use std::error::Error;
use std::fs::File;
use serde::Deserialize;
use ndarray::{Array1, Array2, Axis};
use csv::{WriterBuilder, ReaderBuilder};

const STEPS: u32 = 500;
const RATE: f64 = 0.01;

#[derive(Debug, Deserialize)]
struct Record {
    km: u32,
    price: u32
}


fn load_data(path: &str) -> Result<(Array1<f64>, Array1<f64>), Box<dyn std::error::Error>> {
    let mut features = Vec::new();
    let mut targets = Vec::new();

    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    for result in reader.deserialize() {
        let record: Record = result?;
        targets.push(record.price as f64);
        features.push(record.km as f64);
    }

    let x = Array1::from(features);
    let y = Array1::from(targets);

    Ok((x, y))
}

fn save_thetas(path: &str, thetas: &Array1<f64>) -> Result<(), Box<dyn Error>> {
    let mut writer = WriterBuilder::new().has_headers(false).from_path(path)?;
    writer.serialize(thetas.to_vec())?;

    Ok(())
}

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

fn model(x: &Array2<f64>, thetas: &Array1<f64>) -> Array1<f64> {
    x.dot(thetas)
}

fn cost(x: &Array2<f64>, y: &Array1<f64>, thetas: &Array1<f64>) -> f64 {
    let m = x.len() as f64;
    let p = model(x, thetas);
    let e = &p - y;
    1. / (2. * m) * e.dot(&e)
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = "data.csv";
    let output = "thetas.csv";

    let (mut x, mut y) = load_data(path)?;
    let mut thetas = Array1::zeros(2);

    let (mean_x, std_x) = normalize(&mut x);
    let (mean_y, std_y) = normalize(&mut y);

    let x = add_bias(&x);

    for _ in 0 .. STEPS {
        let predictions = model(&x, &thetas);
        let errors = &predictions - &y;
        let gradients = 1. / x.nrows() as f64 * x.t().dot(&errors);
        thetas = thetas - RATE * gradients;

        let c = cost(&x, &y, &thetas);
        println!("{}", c);
    }

    // Denormalize thetas because trained on normalized values
    thetas[1] = thetas[1] * (std_y / std_x);
    thetas[0] =  (thetas[0] * std_y) + mean_y - thetas[1] * mean_x;

    save_thetas(output, &thetas)?;

    Ok(())
}
