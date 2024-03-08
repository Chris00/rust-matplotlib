use polars::frame::DataFrame;
use polars::series::Series;
use polars::error::PolarsResult;
use polars_core::prelude::*;
use polars::prelude::NamedFrom;
use polars::datatypes::Float64Type;
use matplotlib as plt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (fig, [[mut ax]]) = plt::subplots()?;
    let df = DataFrame::default();
    let pseudacorus_Sepal_Length = Series::new("Pseudacorus_Sepal_Length", &[6.0,6.0,5.8,6.5,5.7,6.8,6.5,6.8,7.0,6.2,6.5,6.6,7.0,7.0,7.5,7.0,6.8,6.5,6.3,6.2,6.9,6.3,6.3,7.0,6.2,6.8,6.2,6.3,5.8,5.9,6.2,6.3,6.9,7.3,7.5,6.4,6.6,6.4,6.7,6.4,6.4,6.6,6.4,6.2,6.3,7.9,6.9,6.2,5.9,6.3]);
    let pseudacorus_Sepal_width = Series::new("Pseudacorus_Sepal_width", &[4.0,3.1,4.0,3.8,3.4,3.7,4.7,4.0,4.5,3.2,3.9,4.0,4.1,4.0,4.6,4.4,4.0,4.2,3.9,4.0,4.8,4.0,4.2,3.8,3.6,3.3,3.4,3.5,3.1,3.7,4.4,4.5,4.4,4.2,4.2,4.4,4.0,4.2,4.0,4.9,4.3,4.4,3.6,3.8,3.5,4.3,4.6,3.8,3.6,3.8]);
   let f: PolarsResult<DataFrame> = DataFrame::new(vec![pseudacorus_Sepal_Length,pseudacorus_Sepal_width]);
   match f {
       Ok(df) => {   
          println!("df is {:?}", &df);
          let x_col = df.column("Pseudacorus_Sepal_Length")?.f64()?;
          let y_col = df.column("Pseudacorus_Sepal_width")?.f64()?;
        
          let x_values: Vec<f64> = x_col.to_vec().iter().map(|&x| x.unwrap()).collect();
          let y_values: Vec<f64> = y_col.to_vec().iter().map(|&y| y.unwrap()).collect();
          println!("x val {:?}", &x_values);
          println!("y val {:?}", &y_values);
          ax.xy(&x_values[..], &y_values[..]).fmt(".").plot();
          fig.save().to_file("flower.png")?;
          }
       Err(e) => {
             println!("error with {:?}", &e);
             }
       }
    Ok(())
}