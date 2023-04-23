use std::{path::Path,
          fmt::{Display, Formatter}};

use pyo3::{prelude::*,
           exceptions::{PyFileNotFoundError, PyPermissionError},
           types::PyTuple, intern};
use numpy::{PyArray1, PyArray2};

macro_rules! py  {
    ($py: ident, $lib: expr, $f: ident) => {
        $lib.getattr($py, intern!($py, stringify!($f))).unwrap()
    };
}

macro_rules! unit_meth {
    ($obj: expr, $m: ident, $py: ident -> $args: expr,
     $e: ident -> $err: expr) => {
        Python::with_gil(|py| {
            let $py = py;
            $obj.call_method1(py, intern!(py, stringify!($m)), $args)
                .map_err(|$e| $err)
        })
    };
    ($obj: expr, $m: ident, $py: ident -> $args: expr) => {
        Python::with_gil(|py| {
            let $py = py;
            $obj.call_method1(py, intern!(py, stringify!($m)), $args)
        })
    };
    ($obj: expr, $m: ident, $args: expr) => {
        Python::with_gil(|py| {
            $obj.call_method1(py, intern!(py, stringify!($m)), $args)
        })
    };
}

/// Possible errors of matplotlib functions.
#[derive(Debug)]
pub enum Error {
    /// Matplotlib was not found on the system.
    NoMatplotlib,
    /// The path contains an elelement that is not a directory or does
    /// not exist.
    FileNotFoundError,
    /// Permission denied to access or create the filesystem path.
    PermissionError,
    /// Unknown error.
    Unknown(PyErr),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Error::NoMatplotlib =>
                write!(f, "the matplotlib library has not been found"),
            Error::FileNotFoundError =>
                write!(f, "A path contains an element that is not a \
                           directory or does not exist"),
            Error::PermissionError =>
                write!(f, "Permission denied to access or create the \
                           filesystem path"),
            Error::Unknown(e) =>
                write!(f, "Unknown Python error: {}", e),
        }
    }
}

impl std::error::Error for Error {}

/// Represent a "connection" to the `numpy` module to be able to
/// perform copy-free conversions of data.
#[derive(Clone)]
pub struct Numpy {
    numpy: Py<PyModule>,
    ctypes: Py<PyModule>,
}

impl Numpy {
    /// Return a new `Numpy`.
    fn new(py: Python) -> Result<Self, Error> {
        let numpy = PyModule::import(py, intern!(py, "numpy.ctypeslib"))
            .map_err(|_| Error::NoMatplotlib)?;
        let ctypes = PyModule::import(py, intern!(py, "ctypes"))
            .map_err(|_| Error::NoMatplotlib)?;
        Ok(Self { numpy: numpy.into(),
                  ctypes: ctypes.into() })
    }
}

/// Trait expressing that `Self` can be converted to a numpy.ndarray
/// (without copying).  `Numpy` is a handle to the numpy library.
pub trait Data {
    fn to_numpy(&self, py: Python, p: &Numpy) -> PyObject;
}

impl<T> Data for T where T: AsRef<[f64]> {
    fn to_numpy(&self, py: Python, p: &Numpy) -> PyObject {
        let x = self.as_ref();
        // ctypes.POINTER(ctypes.c_double)
        let ty = py!(py, p.ctypes, POINTER)
            .call1(py, (py!(py, p.ctypes, c_double),)).unwrap();
        // ctypes.cast(x.as_ptr(), ty)
        let ptr = py!(py, p.ctypes, cast)
            .call1(py, (x.as_ptr() as usize, ty)).unwrap();
        // numpy.ctypeslib.as_array(ptr, shape=(x.len(),))
        py!(py, p.numpy, as_array).call1(py, (ptr, (x.len(),))).unwrap()
    }
}

pub struct Plot {}

#[derive(Clone)]
pub struct Axes {
    axes: PyObject,
    numpy: Numpy,
}

impl Axes {
    fn new(py: Python, axes: PyObject) -> Self {
        Self { axes,  numpy: Numpy::new(py).unwrap() }
    }
}


pub struct Figure {
    plt: Py<PyModule>,
    figure: PyObject,
}

#[inline(always)]
fn grid<const R: usize, const C: usize, U>(
    f: impl Fn(usize, usize) -> U) -> [[U; C]; R] {
    let mut r = 0;
    [(); R].map(|_| {
        let mut c = 0;
        let row = [(); C].map(|_| { let y = f(r, c);
                                    c += 1;
                                    y });
        r += 1;
        row })
}


impl Plot {
    fn pyplot(py: Python) -> Result<&PyModule, Error> {
        PyModule::import(py, intern!(py, "matplotlib.pyplot"))
            .map_err(|_| Error::NoMatplotlib)
    }

    ///
    /// Return an error if Matplotlib is not present on the system.
    pub fn sub<const R: usize, const C: usize>
        () -> Result<(Figure, [[Axes; C]; R]), Error>{
        Python::with_gil(|py| {
            let plt = Self::pyplot(py)?;
            let s = plt.getattr(intern!(py, "subplots")).unwrap();
            let s: &PyTuple = s.call1((R, C)).unwrap().downcast().unwrap();
            let fig = Figure { plt: plt.into(),
                               figure: s.get_item(0).unwrap().into() };
            let axg = s.get_item(1).unwrap();
            let axes;
            if R == 1 {
                if C == 1 {
                    axes = grid(|_,_| Axes::new(py, axg.into()));
                } else { // C > 1
                    let axg: &PyArray1<PyObject> = axg.downcast().unwrap();
                    axes = grid(|_,c| {
                        let ax = axg.get_owned(c).unwrap();
                        Axes::new(py, ax) });
                }
            } else { // R > 1
                if C == 1 {
                    let axg: &PyArray1<PyObject> = axg.downcast().unwrap();
                    axes = grid(|r,_| {
                        let ax = axg.get_owned(r).unwrap();
                        Axes::new(py, ax) });
                } else { // C > 1
                    let axg: &PyArray2<PyObject> = axg.downcast().unwrap();
                    axes = grid(|r, c| {
                        let ax = axg.get_owned([r, c]).unwrap();
                        Axes::new(py, ax) });
                }
            }
            Ok((fig, axes))
        })
    }
}

impl Axes {
    /// Plot the points `x` `y`
    pub fn plot<D>(&mut self, x: &D, y: &D, options: &str) -> &mut Self
    where D: Data + ?Sized {
        // FIXME: Do we want to check that `x` and `y` have the same
        // dimension?  Better error message?
        unit_meth!(self.axes, plot, py -> {
            let xn = x.to_numpy(py, &self.numpy);
            let yn = y.to_numpy(py, &self.numpy);
            (xn, yn, options) })
            .unwrap();
        self
    }

    pub fn scatter<D>(&mut self, x: &D, y: &D) -> &mut Self
    where D: Data + ?Sized {
        // FIXME: Do we want to check that `x` and `y` have the same
        // dimension?  Better error message?
        unit_meth!(self.axes, scatter, py -> {
            let xn = x.to_numpy(py, &self.numpy);
            let yn = y.to_numpy(py, &self.numpy);
            (xn, yn) })
            .unwrap();
        self
    }

    pub fn set_title(&mut self, v: &str) -> &mut Self {
        unit_meth!(self.axes, set_title, (v,)).unwrap();
        self
    }

    /// Set the yaxis' scale.  Possible values for `v` are "linear",
    /// "log", "symlog", "logit",...
    pub fn set_yscale(&mut self, v: &str) -> &mut Self {
        unit_meth!(self.axes, set_yscale, (v,)).unwrap();
        self
    }

    pub fn grid(&mut self) -> &mut Self {
        unit_meth!(self.axes, grid, (true,)).unwrap();
        self
    }

    pub fn set_xlabel(&mut self, label: &str) -> &mut Self {
        unit_meth!(self.axes, set_xlabel, (label,)).unwrap();
        self
    }

    pub fn set_ylabel(&mut self, label: &str) -> &mut Self {
        unit_meth!(self.axes, set_ylabel, (label,)).unwrap();
        self
    }
}

impl Figure {
    // Attach to Plot ??
    pub fn show(self) -> PyResult<()> {
        Python::with_gil(|py| {
            // WARNING: This does not start an envent loop.
            // https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.show
            // self.figure.call_method0(py, "show")?;
            py!(py, self.plt, show).call0(py)?;
            Ok(())
        })
    }

    pub fn savefig(self, path: impl AsRef<Path>) -> Result<(), Error> {
        unit_meth!(self.figure, savefig, py -> (path.as_ref(),), e -> {
            if e.is_instance_of::<PyFileNotFoundError>(py) {
                Error::FileNotFoundError
            } else if e.is_instance_of::<PyPermissionError>(py) {
                Error::PermissionError
            } else {
                Error::Unknown(e)
            } })?;
        Ok(())
    }
}


// pub mod plt {
//     pub fn subplots() -> () {

//     }

// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_basic_pdf() -> Result<(), Error> {
        let (fig, [[mut ax]]) = Plot::sub()?;
        ax.plot(&[1., 2., 3., 4.], &[1., 4., 2., 3.], "");
        fig.savefig("target/a_basic.pdf")?;
        Ok(())
    }

    #[test]
    fn a_basic_row() -> Result<(), Error> {
        let (fig, [[mut ax0, mut ax1]]) = Plot::sub()?;
        ax0.plot(&[1., 2., 3., 4.], &[1., 4., 2., 3.], "");
        ax1.plot(&[1., 2., 3., 4.], &[1., 4., 2., 3.], ".");
        fig.savefig("target/a_basic_row.pdf")?;
        Ok(())
    }

    #[test]
    fn a_basic_col() -> Result<(), Error> {
        let (fig, [[mut ax0], [mut ax1]]) = Plot::sub()?;
        ax0.plot(&[1., 2., 3., 4.], &[1., 4., 2., 3.], "");
        ax1.plot(&[1., 2., 3., 4.], &[1., 4., 2., 3.], ".");
        fig.savefig("target/a_basic_col.pdf")?;
        Ok(())
    }

    #[test]
    fn a_basic_grid() -> Result<(), Error> {
        let (fig, [[mut ax0, mut ax1],
                   [mut ax2, mut ax3]]) = Plot::sub()?;
        ax0.plot(&[1., 2., 3., 4.], &[1., 4., 2., 3.], "");
        ax1.plot(&[1., 2., 3., 4.], &[1., 4., 2., 3.], ".");
        ax2.plot(&[1., 2., 3., 4.], &[1., 4., 2., 3.], "r");
        ax3.plot(&[1., 2., 3., 4.], &[1., 4., 2., 3.], "r.");
        fig.savefig("target/a_basic_grid.pdf")?;
        Ok(())
    }

}
