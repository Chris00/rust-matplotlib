// Example for the README

use matplotlib as plt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (fig, [[mut ax]]) = plt::subplots()?;
    ax.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
    fig.save().to_file("examples/basic_example.svg")?;
    Ok(())
}
