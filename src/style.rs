//! Styles are predefined sets of `rcParams` that define the visual
//! appearance of a plot.
//!
//! [Customizing Matplotlib with style sheets and
//! `rcParams`](https://matplotlib.org/stable/users/explain/customizing.html#customizing) describes
//! the mechanism and usage of styles.

use pyo3::{
    conversion::IntoPyObject,
    Python,
    types::{PyAnyMethods, PyString},
};
use crate::{Error, IntoError, RcParams};

/// Types that may be used as a parameter to [`using`].
pub trait StyleSpec {
    /// Python styles may be specified by `str`, `dict`, `Path` or `list`.
    #[doc(hidden)]
    fn as_py<'py>(&self, py: Python<'py>) -> impl IntoPyObject<'py>;
}

impl StyleSpec for &str {
    fn as_py<'py>(&self, py: Python<'py>) -> impl IntoPyObject<'py> {
        PyString::new(py, self)
    }
}

impl StyleSpec for RcParams {
    fn as_py<'py>(&self, _py: Python<'py>) -> impl IntoPyObject<'py> {
        &self.rc
    }
}


pub fn using(style: impl StyleSpec) -> Result<(), Error> {
    Python::attach(|py| -> Result<_, Error> {
        let style = style.as_py(py);
        py.import("matplotlib.style").unwrap()
            .getattr("use").unwrap()
            .call1((style,))
            .into_error(py)?;
        Ok(())
    })
}
