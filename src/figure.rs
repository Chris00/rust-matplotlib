use crate::axes::Axes;
use crate::{getattr, pyimport, Error, ImportError};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::{
    exceptions::{PyFileNotFoundError, PyPermissionError},
    intern,
    prelude::*,
    types::PyDict,
};
use std::{path::Path, sync::LazyLock};

/// The top level container for all the plot elements.
#[derive(Debug)]
pub struct Figure {
    pub(crate) fig: Py<PyAny>, // instance of matplotlib.figure.Figure
}

/// ⚠ Accessing these may try to lock Python's GIL.  Make sure it is
/// executed outside a call to `Python::attach`.
static FIGURE: LazyLock<Result<Py<PyModule>, ImportError>> =
    LazyLock::new(|| pyimport!(matplotlib::FIGURE, "matplotlib.figure"));

#[inline(always)]
fn grid<const R: usize, const C: usize, U>(f: impl Fn(usize, usize) -> U) -> [[U; C]; R] {
    let mut r = 0;
    [(); R].map(|_| {
        let mut c = 0;
        let row = [(); C].map(|_| {
            let y = f(r, c);
            c += 1;
            y
        });
        r += 1;
        row
    })
}

impl Figure {
    /// Return a new `Figure`.
    ///
    /// ⚠ The figures created with this function will not be displayed
    /// with [`show`].  They can be [saved][Figure::save] to files.
    pub fn new() -> Result<Figure, Error> {
        let figure = FIGURE.as_ref()?;
        Python::attach(|py| {
            let fig = getattr!(py, figure, "Figure").call0(py).unwrap();
            Ok(Self { fig })
        })
    }

    /// Return a grid of subplots with `R` rows and `C` columns.
    pub fn subplots<const R: usize, const C: usize>(&self) -> Result<[[Axes; C]; R], Error> {
        Python::attach(|py| {
            let axs = self.fig.bind(py).call_method1("subplots", (R, C))?;
            let axes;
            if R == 1 {
                if C == 1 {
                    axes = grid(|_, _| Axes {
                        ax: axs.clone().unbind(),
                    });
                } else {
                    // C > 1
                    let axg: &Bound<PyArray1<Py<PyAny>>> = axs.cast().unwrap();
                    axes = grid(|_, c| {
                        let ax = axg.get_owned(c).unwrap();
                        Axes { ax }
                    });
                }
            } else {
                // R > 1
                if C == 1 {
                    let axg: &Bound<PyArray1<Py<PyAny>>> = axs.cast().unwrap();
                    axes = grid(|r, _| {
                        let ax = axg.get_owned(r).unwrap();
                        Axes { ax }
                    });
                } else {
                    // C > 1
                    let axg: &Bound<PyArray2<Py<PyAny>>> = axs.cast().unwrap();
                    axes = grid(|r, c| {
                        let ax = axg.get_owned([r, c]).unwrap();
                        Axes { ax }
                    });
                }
            }
            Ok(axes)
        })
    }

    /// If using a GUI backend with pyplot, display the figure window.
    ///
    /// ⚠ [This does not manage an GUI event loop][GUI]. Consequently,
    /// the figure may only be shown briefly or not shown at all if
    /// you or your environment are not managing an event loop.  Use
    /// [`show()`] for that.
    ///
    /// [GUI]: https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.show
    pub fn show(self) -> Result<(), Error> {
        Python::attach(|py| match self.fig.call_method0(py, intern!(py, "show")) {
            Ok(_) => Ok(()),
            Err(e) => Err(Error::Python(e)),
        })
    }

    /// Save the figure to a file.
    pub fn save(&self) -> Savefig<'_> {
        Savefig {
            fig: &self.fig,
            dpi: None,
        }
    }

    /// Default width: 6.4, default height: 4.8
    pub fn set_size_inches(&mut self, width: f64, height: f64) -> &mut Self {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("size_inches", (width, height)).unwrap();
            self.fig
                .call_method(py, intern!(py, "set"), (), Some(&kwargs))
                .unwrap();
        });
        self
    }
}

/// Options for saving figures.
#[must_use]
pub struct Savefig<'a> {
    fig: &'a Py<PyAny>,
    dpi: Option<f64>,
}

impl<'a> Savefig<'a> {
    pub fn dpi(&mut self, dpi: f64) -> &mut Self {
        if dpi > 0. {
            self.dpi = Some(dpi);
        } else {
            self.dpi = None;
        }
        self
    }

    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            if let Some(dpi) = self.dpi {
                kwargs.set_item("dpi", dpi).unwrap()
            }
            self.fig
                .call_method(py, intern!(py, "savefig"), (path.as_ref(),), Some(&kwargs))
                .map_err(|e| {
                    if e.is_instance_of::<PyFileNotFoundError>(py) {
                        Error::FileNotFoundError
                    } else if e.is_instance_of::<PyPermissionError>(py) {
                        Error::PermissionError
                    } else {
                        Error::Python(e)
                    }
                })
        })?;
        Ok(())
    }
}
