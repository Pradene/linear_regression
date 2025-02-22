use csv::ReaderBuilder;
use ndarray::Array1;
use std::error::Error;
use std::io;

fn load_thetas_structured(path: &str) -> Result<Array1<f64>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(false).from_path(path)?;

    let record: Vec<f64> = match reader.deserialize().next() {
        Some(result) => result?, // Handle potential deserialization errors
        None => return Err("CSV file is empty".into()),
    };
    
    Ok(Array1::from_vec(record))
}

fn predict(x: f64, theta0: f64, theta1: f64) -> f64 {
    theta0 + x * theta1
}

fn main() -> Result<(), Box<dyn Error>> {
    let thetas = load_thetas_structured("thetas.csv")?;
    let theta0 = &thetas[0];
    let theta1 = &thetas[1];

    println!("What is the mileage of your car ?");

    let mut input = String::new();
    io::stdin().read_line(&mut input)
        .expect("Failed to read input");

    let trimmed = input.trim();
    match trimmed.parse::<f64>() {
        Ok(x) => println!("The estimated price of your car is : {}", predict(x, *theta0, *theta1)),
        Err(_) => println!("Invalid input! Not a valid number"),
    }

    Ok(())
}