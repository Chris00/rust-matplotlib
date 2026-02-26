//! [Rust][] bindings to [Matplotlib][] Python visualization library.
//!
//! Usage
//! -----
//!
//! These bindings provide an interface close to [Matplotlib][]'s explicit
//! one while keeping a Rust flavor.
//!
//! [Rust]: https://www.rust-lang.org/
//! [Matplotlib]: https://matplotlib.org/

use std::{
    fmt::{Display, Formatter},
    sync::LazyLock,
};
use pyo3::{
    prelude::*,
    intern,
};

pub mod colors;
pub mod figure;
use figure::Figure;
pub mod axes;
use axes::Axes;
pub mod lines;
mod macros;

/// Possible errors of matplotlib functions.
#[derive(Debug)]
pub enum Error {
    /// The Python library "matplotlib" was not found.
    NoMatplotlib(String),
    /// The path contains an element that is not a directory or does
    /// not exist.
    FileNotFoundError,
    /// Permission denied to access or create the filesystem path.
    PermissionError,
    /// Other Python errors.
    Python(PyErr),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Error::NoMatplotlib(e) =>
                write!(f, "The matplotlib library has not been found.\n\
Python Error: {e}\n\
Please install it.  See https://matplotlib.org\n\
If you use Anaconda, see https://github.com/PyO3/pyo3/issues/1554"),
            Error::FileNotFoundError =>
                write!(f, "A path contains an element that is not a \
                           directory or does not exist"),
            Error::PermissionError =>
                write!(f, "Permission denied to access or create the \
                           filesystem path"),
            Error::Python(e) =>
                write!(f, "Python error: {}", e),
        }
    }
}

impl std::error::Error for Error {}

impl From<PyErr> for Error {
    fn from(e: PyErr) -> Self {
        Error::Python(e)
    }
}

#[derive(Debug)]
struct ImportError(String);

impl From<&ImportError> for Error {
    fn from(e: &ImportError) -> Self {
        Error::NoMatplotlib(e.0.clone())
    }
}

/// ⚠ Accessing these may try to lock Python's GIL.  Make sure it is
/// executed outside a call to `Python::attach`.
static PYPLOT: LazyLock<Result<Py<PyModule>, ImportError>> =
    LazyLock::new(|| {
        pyimport!(matplotlib::PYPLOT, "matplotlib.pyplot")
    });

// RuntimeWarning: More than 20 figures have been opened. Figures
// created through the pyplot interface (`matplotlib.pyplot.figure`)
// are retained until explicitly closed and may consume too much
// memory. […]  Consider using `matplotlib.pyplot.close()`.
//
// => Do not use pyplot interface (since we need handles anyway).


/// Return a new figure.
/// This figure is tracked by Matplotlib so [`show()`] displays it.
/// This implies it must be explicitly deallocated using [`close`].
pub fn figure() -> Result<Figure, Error> {
    let pyplot = PYPLOT.as_ref()?;
    Python::attach(|py| {
        let fig = getattr!(py, pyplot, "figure").call0(py)?;
        Ok(Figure { fig })
    })
}

/// Return a figure and a grid of subplots with `R` rows and `C` columns.
pub fn subplots<const R: usize, const C: usize>(
) -> Result<(Figure, [[Axes; C]; R]), Error> {
    let fig = figure()?;
    let ax = fig.subplots()?;
    Ok((fig, ax))
}

/// Display all open figures created with [`figure`] or [`subplots`].
pub fn show() {
    let pyplot = PYPLOT.as_ref().unwrap();
    Python::attach(|py| {
        // FIXME: What do we want to do with the errors?
        getattr!(py, pyplot, "show").call0(py).unwrap();
    })
}

/// Close the figure `fig` (created with [`figure`] or [`subplots`]).
pub fn close(fig: Figure) {
    let pyplot = PYPLOT.as_ref().unwrap();
    Python::attach(|py| {
        getattr!(py, pyplot, "close").call1(py, (fig.fig,)).unwrap();
    })
}

/// Close all figures created with [`figure`] or [`subplots`].
pub fn close_all() {
    let pyplot = PYPLOT.as_ref().unwrap();
    Python::attach(|py| {
        getattr!(py, pyplot, "close").call1(py, ("all",)).unwrap();
    })
}


#[cfg(test)]
mod tests {
    use super::{*, figure::Figure};

    #[test]
    fn a_basic_pdf() -> Result<(), Error> {
        let fig = Figure::new()?;
        let [[mut ax]] = fig.subplots()?;
        dbg!(&fig);
        ax.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
        fig.save().to_file("target/a_basic.pdf")?;
        Ok(())
    }

    #[test]
    fn a_basic_label() -> Result<(), Error> {
        let fig = Figure::new()?;
        let [[mut ax]] = fig.subplots()?;
        dbg!(&fig);
        ax.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.])
            .label("first").plot();
        ax.xy(&[1., 2., 3., 4.], &[4., 2., 3., 1.])
            .label("second".to_string()).plot();
        ax.legend([]);
        fig.save().to_file("target/a_basic_label.pdf")?;
        Ok(())
    }

    #[test]
    fn a_basic_row() -> Result<(), Error> {
        let fig = Figure::new()?;
        let [[mut ax0, mut ax1]] = fig.subplots()?;
        ax0.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
        ax1.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt(".").plot();
        fig.save().to_file("target/a_basic_row.pdf")?;
        Ok(())
    }

    #[test]
    fn a_basic_col() -> Result<(), Error> {
        let fig = Figure::new()?;
        let [[mut ax0], [mut ax1]] = fig.subplots()?;
        ax0.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
        ax1.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt(".").plot();
        fig.save().to_file("target/a_basic_col.pdf")?;
        Ok(())
    }

    #[test]
    fn a_basic_grid() -> Result<(), Error> {
        let fig = Figure::new()?;
        let [[mut ax0, mut ax1],
             [mut ax2, mut ax3]] = fig.subplots()?;
        ax0.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
        ax1.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt(".").plot();
        ax2.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt("r").plot();
        ax3.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt("r.").plot();
        fig.save().to_file("target/a_basic_grid.pdf")?;
        Ok(())
    }

    #[test]
    //#[compile_fail]
    fn data_in_scope() -> Result<(), Error> {
        let fig = Figure::new()?;
        let [[mut ax]] = fig.subplots()?;
        // let l = ax.y(&vec![1., 2.]);
        // l.plot();
        ax.y(&vec![1., 2.]).plot();
        fig.save().to_file("target/data_in_scope.pdf")?;
        Ok(())
    }

}

#[cfg(doctest)]
doc_comment::doctest!("../README.md");
