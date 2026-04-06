//! https://matplotlib.org/stable/tutorials/introductory/quick_start.html#a-simple-example

use matplotlib::pyplot as plt;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let (fig, [[mut ax]]) = plt::subplots()?;
    let x: Vec<_> = (0..1000).map(|i| i as f64 / 10.).collect();
    let y: Vec<_> = x.iter().map(|x| x.sin()).collect();
    ax.xy(&x, &y).fmt("r.").plot();
    // plt::show();
    fig.save().to_file("target/a_simple_example.svg")?;
    Ok(())
}
