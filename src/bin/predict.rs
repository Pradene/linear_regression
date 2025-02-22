use std::error::Error;
use std::io;
use ndarray::{Array1, Axis};

use linear_regression::{load_thetas, model};

fn main() -> Result<(), Box<dyn Error>> {
    let thetas = load_thetas()?;

    println!("What is the mileage of your car ?");

    let mut input = String::new();
    io::stdin().read_line(&mut input)
        .expect("Failed to read input");

    let trimmed: &str = input.trim();
    match trimmed.parse::<f64>() {
        Ok(x) => {
            let x = Array1::from_vec(vec![1., x]).insert_axis(Axis(0));
            println!("The estimated price of your car is : {}", model(&x, &thetas));
        },
        Err(_) => println!("Invalid input! Not a valid number"),
    }

    Ok(())
}
