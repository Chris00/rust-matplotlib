// Inspired by https://gitlab.com/whooie/mpl/-/blob/master/README.md?ref_type=heads#example

use matplotlib::{self as mpl, pyplot as plt};
use std::{error, f64::consts::TAU};

fn main() -> Result<(), Box<dyn error::Error>> {
    // Remark: `ax.fun` is easier (and more efficient) for plotting functions.
    let dx: f64 = TAU / 50.0;
    let x = (0..50_u32).map(|k| f64::from(k) * dx);
    let y1 = x.clone().map(f64::sin);
    let y2 = x.clone().map(f64::cos);

    mpl::rc_params().set("axes.linewidth", 0.65)?;
    mpl::rc_params().set("lines.linewidth", 0.8)?;
    let (_, [[mut ax]]) = plt::subplots()?;
    ax.grid().set_xlabel("$x$");
    ax.xy_from(x.clone().zip(y1))
        .fmt("ob-")
        .label("$\\sin(x)$")
        .plot();
    ax.xy_from(x.zip(y2)).fmt("Dr-").label("$\\cos(x)$").plot();
    ax.legend([]);
    plt::show();

    Ok(())
}
