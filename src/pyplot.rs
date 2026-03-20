//! State-based interface to matplotlib.  It provides an implicit,
//! MATLAB-like, way of plotting.  It also opens figures on your
//! screen, and acts as the figure GUI manager.
//!
//! pyplot is mainly intended for interactive plots and simple cases
//! of programmatic plot generation:
//!
//! ```no_run
//! use matplotlib::pyplot as plt;
//! let (_, [[mut ax]]) = plt::subplots()?;
//! ax.fun(f64::sin, 0., 5.).plot();
//! plt::show();
//! # Ok::<(), matplotlib::Error>(())
//! ```

use pyo3::{
    prelude::*, sync::PyOnceLock,
};
use crate::{axes::Axes, Error, figure::Figure, IntoError};

// RuntimeWarning: More than 20 figures have been opened. Figures
// created through the pyplot interface (`matplotlib.pyplot.figure`)
// are retained until explicitly closed and may consume too much
// memory. […]  Consider using `matplotlib.pyplot.close()`.
//
// => Do not use pyplot interface (since we need handles anyway).

/// Return a new figure.
/// This figure is tracked by Matplotlib so [`show()`] displays it.
/// This implies it must be explicitly deallocated using [`close()`].
pub fn figure() -> Result<Figure, Error> {
    static FIG: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    Python::attach(|py| {
        let fig = FIG.import(py, "matplotlib.pyplot", "figure")
            .into_error(py)?;
        Ok(Figure { fig: fig.call0().into_error(py)?.unbind() })
    })
}

/// Return a figure and a grid of subplots with `R` rows and `C` columns.
pub fn subplots<const R: usize, const C: usize>(
) -> Result<(Figure, [[Axes; C]; R]), Error> {
    let fig = figure()?;
    let ax = fig.subplots()?;
    Ok((fig, ax))
}

/// Display all open figures created with [`figure()`] or [`subplots()`].
pub fn show() {
    static SHOW: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    Python::attach(|py| {
        SHOW.import(py, "matplotlib.pyplot", "show")
            .expect("Cannot get matplotlib.pyplot.show")
            .call0()
            .expect("matplotlib.pyplot.show did not succeed");
    })
}

/// Close the figure `fig` (created with [`figure()`] or [`subplots()`]).
pub fn close(fig: Figure) {
    static CLOSE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    Python::attach(|py| {
        CLOSE.import(py, "matplotlib.pyplot", "close")
            .expect("Cannot find matplotlib.pyplot.close")
            .call1((fig.fig,))
            .expect("matplotlib.pyplot.close did not succeed");
    })
}

/// Close all figures created with [`figure()`] or [`subplots()`].
pub fn close_all() {
    static CLOSE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    Python::attach(|py| {
        CLOSE.import(py, "matplotlib.pyplot", "close")
            .expect("Cannot get matplotlib.pyplot.close")
            .call1(("all",))
            .expect("matplotlib.pyplot.close('all') did not succeed");
    })
}
