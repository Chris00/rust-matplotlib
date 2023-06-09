/// https://matplotlib.org/stable/tutorials/introductory/quick_start.html#a-simple-example

use std::error::Error;
use matplotlib::*;

fn main() -> Result<(), Box<dyn Error>> {
    let (fig, [[mut ax]]) = Plot::sub()?;
    let x: Vec<_> = (0 .. 1000).map(|i| i as f64 / 10.).collect();
    let y: Vec<_> = x.iter().map(|x| x.sin()).collect();
    ax.xy(&x, &y).fmt("r.").plot();
    // fig.show();
    fig.savefig("/tmp/a_simple_example.pdf")?;
    Ok(())
}
