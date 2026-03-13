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
};
use pyo3::{
    exceptions::{PyFileNotFoundError, PyPermissionError},
    intern, prelude::*, sync::PyOnceLock,
};

pub mod colors;
pub mod figure;
use figure::Figure;
pub mod axes;
use axes::Axes;
pub mod lines;
pub mod text;
include!("macros.rs");

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

/// Conversion from `PyErr` to `Error` (requires the Python handle).
trait IntoError {
    type T;
    fn into_error(self, py: Python<'_>) -> Result<Self::T, Error>;
}

impl<T> IntoError for Result<T, PyErr> {
    type T = T;

    fn into_error(self, py: Python<'_>) -> Result<T, Error> {
        self.map_err(|e| {
            if e.is_instance_of::<PyFileNotFoundError>(py) {
                Error::FileNotFoundError
            } else if e.is_instance_of::<PyPermissionError>(py) {
                Error::PermissionError
            } else {
                Error::Python(e)
            }
        })
    }
}

#[derive(Debug)]
struct ImportError(String);

impl From<&ImportError> for Error {
    fn from(e: &ImportError) -> Self {
        Error::NoMatplotlib(e.0.clone())
    }
}

// RuntimeWarning: More than 20 figures have been opened. Figures
// created through the pyplot interface (`matplotlib.pyplot.figure`)
// are retained until explicitly closed and may consume too much
// memory. […]  Consider using `matplotlib.pyplot.close()`.
//
// => Do not use pyplot interface (since we need handles anyway).

fn pyplot<'a, 'py>(
    py: Python<'py>,
    f: &'a PyOnceLock<Py<PyAny>>,
    attr_name: &str
) -> Result<&'a Bound<'py, PyAny>, Error> {
    f.get_or_try_init(py, || {
        Ok(py.import("matplotlib.pyplot").into_error(py)?
            .getattr(attr_name).into_error(py)?
            .unbind())
    })
        .map(|f| f.bind(py))
}

/// Return a new figure.
/// This figure is tracked by Matplotlib so [`show()`] displays it.
/// This implies it must be explicitly deallocated using [`close()`].
pub fn figure() -> Result<Figure, Error> {
    static FIG: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    Python::attach(|py| {
        let fig = pyplot(py, &FIG, "figure")?;
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
        pyplot(py, &SHOW, "show")
            .expect("Cannot get matplotlib.pyplot.show")
            .call0()
            .expect("matplotlib.pyplot.show did not succeed");
    })
}

/// Close the figure `fig` (created with [`figure()`] or [`subplots()`]).
pub fn close(fig: Figure) {
    static CLOSE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    Python::attach(|py| {
        pyplot(py, &CLOSE, "close")
            .expect("Cannot find matplotlib.pyplot.close")
            .call1((fig.fig,))
            .expect("matplotlib.pyplot.close did not succeed");
    })
}

/// Close all figures created with [`figure()`] or [`subplots()`].
pub fn close_all() {
    static CLOSE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    Python::attach(|py| {
        pyplot(py, &CLOSE, "close")
            .expect("Cannot get matplotlib.pyplot.close")
            .call1(("all",))
            .expect("matplotlib.pyplot.close('all') did not succeed");
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
