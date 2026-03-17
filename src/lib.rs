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
    exceptions::{PyFileNotFoundError, PyValueError, PyPermissionError},
    intern,
    prelude::*,
    types::PyDict,
    sync::PyOnceLock,
};

pub mod colors;
pub mod figure;
pub mod axes;
pub mod lines;
pub mod pyplot;
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
    /// Indicate that the argument is an inappropriate value.
    ValueError(String),
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
            Error::ValueError(msg) =>
                write!(f, "ValueError: {}", msg),
            Error::Python(e) =>
                write!(f, "Python error: {}", e),
        }
    }
}

impl std::error::Error for Error {}

/// Conversion from `PyErr` to `Error`.
// It requires the Python handle, so the `Into` trait cannot be used.
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
            } else if e.is_instance_of::<PyValueError>(py) {
                let msg = e.value(py).str().unwrap();
                Error::ValueError(msg.to_string_lossy().into_owned())
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

/// A dict-like key-value store for config parameters.
#[derive(Debug)]
pub struct RcParams {
    rc: Py<PyDict>,
}

/// Rust types that may be used as [`RcParams`] values.
pub trait RcParamsValue {
    fn into_py<'py>(&self, py: Python<'py>) -> impl IntoPyObject<'py>;
}

impl RcParamsValue for &str {
    fn into_py<'py>(&self, _py: Python<'py>) -> impl IntoPyObject<'py> {
        self
    }
}

impl RcParamsValue for usize {
    fn into_py<'py>(&self, _py: Python<'py>) -> impl IntoPyObject<'py> {
        self
    }
}

impl RcParamsValue for f64 {
    fn into_py<'py>(&self, _py: Python<'py>) -> impl IntoPyObject<'py> {
        self
    }
}

// TODO: Make more types implement `RcParamsValue`.

impl RcParams {
    pub fn get(&self, key: &str) -> Option<Py<PyAny>> {
        Python::attach(|py| {
            self.rc.bind(py).get_item(key)
                .unwrap()
                .map(|o| o.unbind())
        })
    }

    /// Set the [RcParam][] `key` to `value`.
    ///
    /// Note that an improper value may only raise an error later such
    /// as when [`pyplot::figure`] is called.
    ///
    /// [RcParam]: https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams
    pub fn set<'py>(&self, key: &str, value: impl RcParamsValue) -> Result<(), Error> {
        Python::attach(|py| -> Result<_, Error> {
            self.rc.bind(py).set_item(key, value.into_py(py))
                .into_error(py)
        })
    }

    /// Return the subset of the `self` dictionary whose keys match
    /// ([`using
    /// re.search()`](https://docs.python.org/3/library/re.html#re.search)) the
    /// given pattern.
    pub fn find_all(&self, pat: &str) -> Vec<String> {
        Python::attach(|py| {
            self.rc.bind(py)
                .call_method1(intern!(py, "find_all"), (pat,)).unwrap()
                .cast().unwrap()
                .extract().unwrap()
        })
    }
}

/// Return an instance of [`RcParams`] for handling default Matplotlib
/// values.
///
/// Setting values must be done before the figure is created.
pub fn rc_params() -> &'static RcParams {
    static RCPARAMS: PyOnceLock<RcParams> = PyOnceLock::new();
    Python::attach(|py| {
        RCPARAMS.get_or_init(py, || {
            let rc = py.import("matplotlib")
                .expect("Cannot find matplotlib")
                .getattr("rcParams")
                .expect("Cannot find matplotlib.rcParams")
                .cast_into::<PyDict>().unwrap();
            RcParams { rc: rc.unbind() }
        });
        RCPARAMS.get(py).unwrap()
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
            .label(&"second".to_string()).plot();
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
