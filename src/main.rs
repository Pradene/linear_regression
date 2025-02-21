use std::error::Error;
use std::fs::File;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Record {
    km: String,
    price: String
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut records: Vec<Record> = Vec::new();

    let path = "data.csv";
    let file = File::open(path)?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    for result in reader.deserialize() {
        let record: Record = result?;
        records.push(record);
    }

    println!("{:#?}", records);

    Ok(())
}