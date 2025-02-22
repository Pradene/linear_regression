use ndarray::{Array1, Array2};
use serde::Deserialize;
use csv::{ReaderBuilder, WriterBuilder};
use std::fs::File;
use std::error::Error;

pub const DATA_FILE: &str = "data.csv";
pub const THETAS_FILE: &str = "thetas.csv";

pub fn model(x: &Array2<f64>, thetas: &Array1<f64>) -> Array1<f64> {
    x.dot(thetas)
}

#[derive(Debug, Deserialize)]
pub struct Record {
    pub km: u32,
    pub price: u32
}

pub fn load_data() -> Result<(Array1<f64>, Array1<f64>), Box<dyn std::error::Error>> {
    let mut features = Vec::new();
    let mut targets = Vec::new();

    let file = File::open(DATA_FILE)?;
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

pub fn save_thetas(thetas: &Array1<f64>) -> Result<(), Box<dyn Error>> {
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .from_path(THETAS_FILE)?;

    writer.serialize(thetas.to_vec())?;

    Ok(())
}

pub fn load_thetas() -> Result<Array1<f64>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(THETAS_FILE)?;

    let record: Vec<f64> = match reader.deserialize().next() {
        Some(result) => result?, // Handle potential deserialization errors
        None => return Err("CSV file is empty".into()),
    };
    
    Ok(Array1::from_vec(record))
}