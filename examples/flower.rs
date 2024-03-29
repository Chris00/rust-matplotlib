use polars_core::prelude::*;
use matplotlib as plt;

// Yellow Flag Iris, attributed to Iris pseudacorus flowers were
// measured in Norfolk marshes in 2018 by LCrossman.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (fig, [[mut ax]]) = plt::subplots()?;
    let pseudacorus_sepal_length = Series::new("Pseudacorus_Sepal_Length",
        &[6.0,6.0,5.8,6.5,5.7,6.8,6.5,6.8,7.0,6.2,6.5,6.6,7.0,7.0,7.5,7.0,
            6.8,6.5,6.3,6.2,6.9,6.3,6.3,7.0,6.2,6.8,6.2,6.3,5.8,5.9,6.2,6.3,
            6.9,7.3,7.5,6.4,6.6,6.4,6.7,6.4,6.4,6.6,6.4,6.2,6.3,7.9,6.9,6.2,
            5.9,6.3]);
    let pseudacorus_sepal_width = Series::new("Pseudacorus_Sepal_width",
        &[4.0,3.1,4.0,3.8,3.4,3.7,4.7,4.0,4.5,3.2,3.9,4.0,4.1,4.0,4.6,4.4,
            4.0,4.2,3.9,4.0,4.8,4.0,4.2,3.8,3.6,3.3,3.4,3.5,3.1,3.7,4.4,4.5,
            4.4,4.2,4.2,4.4,4.0,4.2,4.0,4.9,4.3,4.4,3.6,3.8,3.5,4.3,4.6,3.8,
            3.6,3.8]);
    let df = DataFrame::new(
        vec![pseudacorus_sepal_length,pseudacorus_sepal_width])?;
    println!("df is {:?}", &df);
    let x_col = df.column("Pseudacorus_Sepal_Length")?.f64()?;
    let y_col = df.column("Pseudacorus_Sepal_width")?.f64()?;

    ax.xy_from(x_col.iter().zip(y_col)).fmt(".").plot();
    fig.save().to_file("flower.png")?;
    Ok(())
}
