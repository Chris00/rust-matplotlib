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
    borrow::Borrow,
    fmt::{Display, Formatter},
    mem::swap,
    path::Path,
};
use lazy_static::lazy_static;
use pyo3::{
    prelude::*,
    intern,
    exceptions::{PyFileNotFoundError, PyPermissionError},
    types::{IntoPyDict, PyDict, PyList},
};
use numpy::{
    PyArray1,
    PyArray2,
};

#[cfg(feature = "curve-sampling")]
use curve_sampling::Sampling;

macro_rules! getattr {
    ($py: ident, $lib: expr, $f: literal) => {
        $lib.getattr($py, intern!($py, $f)).unwrap()
    };
}

macro_rules! meth {
    ($obj: expr, $m: ident, $py: ident -> $args: expr,
     $e: ident -> $err: expr) => {
        Python::with_gil(|py| {
            let $py = py;
            $obj.call_method1(py, intern!(py, stringify!($m)), $args)
                .map_err(|$e| $err)
        })
    };
    ($obj: expr, $m: ident, $py: ident -> $args: expr, $kwargs: expr) => {
        Python::with_gil(|py| {
            let $py = py;
            $obj.call_method(py, intern!(py, stringify!($m)), $args, $kwargs)
        })
    };
    ($obj: expr, $m: ident, $py: ident -> $args: expr) => {
        Python::with_gil(|py| {
            let $py = py;
            $obj.call_method1(py, intern!(py, stringify!($m)), $args)
        })
    };
    ($obj: expr, $m: ident, $args: expr, $kwargs: expr) => {
        Python::with_gil(|py| {
            $obj.call_method(py, intern!(py, stringify!($m)), $args, $kwargs)
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
    /// The Python library "matplotlib" was not found.
    NoMatplotlib,
    /// The path contains an elelement that is not a directory or does
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
            Error::NoMatplotlib =>
                write!(f, "The matplotlib library has not been found.\n\
Please install it.  See https://matplotlib.or/\n\
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

/// Import and return a handle to the module `$m`.
macro_rules! pyimport { ($m: literal) => {
    Python::with_gil(|py|
        PyModule::import(py, intern!(py, $m)).map(|m| m.into()))
}}

lazy_static! {
    // Import matplotlib modules.
    static ref FIGURE: Result<Py<PyModule>, PyErr> = {
        pyimport!("matplotlib.figure")
    };
    static ref PYPLOT: Result<Py<PyModule>, PyErr> = {
        pyimport!("matplotlib.pyplot")
    };
    static ref NUMPY: Result<Numpy, PyErr> = {
        Ok(Numpy {
            numpy: pyimport!("numpy.ctypeslib")?,
            ctypes: pyimport!("ctypes")?,
        })
    };
}

/// Return a handle to the module `$m`.
/// ⚠ This may try to lock Python's GIL.  Make sure it is executed
/// outside a call to `Python::with_gil`.
macro_rules! pymod { ($m: ident) => {
    $m.as_ref().map_err(|_| Error::NoMatplotlib)
}}


/// Represent a "connection" to the `numpy` module to be able to
/// perform copy-free conversions of data.
#[derive(Clone)]
pub struct Numpy {
    numpy: Py<PyModule>,
    ctypes: Py<PyModule>,
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
        let ty = getattr!(py, p.ctypes, "POINTER")
            .call1(py, (getattr!(py, p.ctypes, "c_double"),)).unwrap();
        // ctypes.cast(x.as_ptr(), ty)
        let ptr = getattr!(py, p.ctypes, "cast")
            .call1(py, (x.as_ptr() as usize, ty)).unwrap();
        // numpy.ctypeslib.as_array(ptr, shape=(x.len(),))
        getattr!(py, p.numpy, "as_array")
            .call1(py, (ptr, (x.len(),))).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct Axes {
    ax: PyObject,
}

/// The top level container for all the plot elements.
#[derive(Debug)]
pub struct Figure {
    fig: PyObject, // instance of matplotlib.figure.Figure
}

pub struct Line2D {
    line2d: Py<PyList>,
}

#[inline(always)]
fn grid<const R: usize, const C: usize, U>(
    f: impl Fn(usize, usize) -> U) -> [[U; C]; R] {
    let mut r = 0;
    [(); R].map(|_| {
        let mut c = 0;
        let row = [(); C].map(|_| {
            let y = f(r, c);
            c += 1;
            y });
        r += 1;
        row })
}

impl Figure {
    /// Return a new `Figure`.
    pub fn new() -> Result<Figure, Error> {
        let figure = pymod!(FIGURE)?;
        Python::with_gil(|py| {
            let fig = getattr!(py, figure, "Figure")
                .call0(py).unwrap();
            Ok(Self { fig: fig.into() })
        })
    }

    ///
    /// Return an error if Matplotlib is not present on the system.
    pub fn subplots<const R: usize, const C: usize>(
        &self) -> Result<[[Axes; C]; R], Error> {
        Python::with_gil(|py| {
            let axs = self.fig
                .call_method1(py, "subplots", (R, C))
                .map_err(|e| Error::Python(e))?;
            let axes;
            if R == 1 {
                if C == 1 {
                    axes = grid(|_,_| Axes { ax: axs.clone() });
                } else { // C > 1
                    let axg: &PyArray1<PyObject> = axs.downcast(py).unwrap();
                    axes = grid(|_,c| {
                        let ax = axg.get_owned(c).unwrap();
                        Axes { ax } });
                }
            } else { // R > 1
                if C == 1 {
                    let axg: &PyArray1<PyObject> = axs.downcast(py).unwrap();
                    axes = grid(|r,_| {
                        let ax = axg.get_owned(r).unwrap();
                        Axes { ax } });
                } else { // C > 1
                    let axg: &PyArray2<PyObject> = axs.downcast(py).unwrap();
                    axes = grid(|r, c| {
                        let ax = axg.get_owned([r, c]).unwrap();
                        Axes { ax } });
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
    /// [`matplotlib::show()`] for that.
    ///
    /// [GUI]: https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.show
    pub fn show(self) -> Result<(), Error> {
        Python::with_gil(|py|
            match self.fig.call_method0(py, intern!(py, "show")) {
                Ok(_) => Ok(()),
                Err(e) => Err(Error::Python(e)),
            })
    }

    pub fn save(&self) -> Savefig {
        Savefig { fig: self.fig.clone(), dpi: None }
    }
}

pub struct Savefig {
    fig: PyObject,
    dpi: Option<f64>,
}

impl Savefig {
    pub fn dpi(&mut self, dpi: f64) -> &mut Self {
        if dpi > 0. {
            self.dpi = Some(dpi);
        } else {
            self.dpi = None;
        }
        self
    }

    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            if let Some(dpi) = self.dpi {
                kwargs.set_item("dpi", dpi).unwrap()
            }
            self.fig.call_method(
                py, intern!(py, "savefig"),
                (path.as_ref(),), Some(kwargs)
            ).map_err(|e| {
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


pub fn figure() -> Result<Figure, Error> {
    let pyplot = pymod!(PYPLOT)?;
    Python::with_gil(|py| {
        let fig = getattr!(py, pyplot, "figure")
            .call0(py).map_err(|e| Error::Python(e))?;
        Ok(Figure { fig: fig.into() })
    })
}

pub fn subplots<const R: usize, const C: usize>(
) -> Result<(Figure, [[Axes; C]; R]), Error> {
    let fig = figure()?;
    let ax = fig.subplots()?;
    Ok((fig, ax))
}

/// Display all open figures.
pub fn show() {
    let pyplot = pymod!(PYPLOT).unwrap();
    Python::with_gil(|py| {
        // FIXME: What do we want to do with the errors?
        getattr!(py, pyplot, "show").call0(py).unwrap();
    })
}


impl Axes {
    /// Plot `y` versus `x` as lines and/or markers.
    ///
    /// # Example
    ///
    /// ```
    /// use matplotlib as plt;
    /// let (fig, [[mut ax]]) = plt::subplots()?;
    /// ax.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
    /// fig.save().to_file("target/XY_plot.pdf")?;
    /// # Ok::<(), matplotlib::Error>(())
    /// ```
    // FIXME: Do we want to check that `x` and `y` have the same
    // dimension?  Better error message?
    #[must_use]
    pub fn xy<'a, D>(&'a mut self, x: &'a D, y: &'a D) -> XY<'a, D>
    where D: Data + ?Sized {
        // The chain leading to plot starts with the data (using this
        // function) so that additional data may be added, sharing
        // common options.  We also mutably borrow `self` to reflect that
        // the final `.plot()` will mutate the underlying Python object.
        XY { axes: self,
             options: PlotOptions::new(),
             data: PlotData::XY(x, y),
             prev_data: vec![] }
    }

    /// Plot `y` versus its indices as lines and/or markers.
    ///
    /// # Example
    ///
    /// ```
    /// use matplotlib as plt;
    /// let (fig, [[mut ax]]) = plt::subplots()?;
    /// ax.y(&[1., 4., 2., 3.]).plot();
    /// fig.save().to_file("target/Y_plot.pdf")?;
    /// # Ok::<(), matplotlib::Error>(())
    /// ```
    #[must_use]
    pub fn y<'a, D>(&'a mut self, y: &'a D) -> XY<'a, D>
    where D: Data + ?Sized {
        XY { axes: self,
             options: PlotOptions::new(),
             data: PlotData::Y(y),
             prev_data: vec![] }
    }

    /// Convenience function to plot X-Y coordinates coming from `xy`.
    ///
    /// # Example
    ///
    /// ```
    /// use matplotlib as plt;
    /// let (fig, [[mut ax]]) = plt::subplots()?;
    /// ax.xy_from(&[(1., 2.), (4., 2.), (2., 3.), (3., 4.)]).plot();
    /// ax.xy_from([(1., 0.), (2., 3.), (3., 1.), (4., 3.)]).plot();
    /// fig.save().to_file("target/XY_from_plot.pdf")?;
    /// # Ok::<(), matplotlib::Error>(())
    /// ```
    #[must_use]
    pub fn xy_from<'a, I>(&'a mut self, xy: I) -> XYFrom<'a, I>
    where I: IntoIterator,
          <I as IntoIterator>::Item: Borrow<(f64, f64)> {
        // (f64, f64) chosend for comatibility with `zip`.
        XYFrom { axes: self,
                 options: PlotOptions::new(),
                 data: xy }
    }

    #[cfg(feature = "curve-sampling")]
    /// Plot the graph of the function `f` on the interval \[`a`, `b`\].
    ///
    /// # Example
    /// ```
    /// use matplotlib as plt;
    /// let (fig, [[mut ax]]) = plt::subplots()?;
    /// ax.fun(|x| x * x, 0., 1.).plot();
    /// fig.save().to_file("target/Fun_plot.pdf")?;
    /// # Ok::<(), matplotlib::Error>(())
    /// ```
    pub fn fun<'a, F>(&'a mut self, f: F, a: f64, b: f64) -> Fun<'a, F>
    where F: FnMut(f64) -> f64 {
        Fun { axes: self,
              options: PlotOptions::new(),
              f, a, b,
              n: 100 }
    }


    #[must_use]
    pub fn scatter<D>(&mut self, x: &D, y: &D) -> &mut Self
    where D: Data + ?Sized {
        // FIXME: Do we want to check that `x` and `y` have the same
        // dimension?  Better error message?
        let numpy = pymod!(NUMPY).unwrap();
        meth!(self.ax, scatter, py -> {
            let xn = x.to_numpy(py, &numpy);
            let yn = y.to_numpy(py, &numpy);
            (xn, yn) })
            .unwrap();
        self
    }

    pub fn set_title(&mut self, v: &str) -> &mut Self {
        meth!(self.ax, set_title, (v,)).unwrap();
        self
    }

    /// Set the yaxis' scale.  Possible values for `v` are "linear",
    /// "log", "symlog", "logit",...
    pub fn set_yscale(&mut self, v: &str) -> &mut Self {
        meth!(self.ax, set_yscale, (v,)).unwrap();
        self
    }

    pub fn grid(&mut self) -> &mut Self {
        meth!(self.ax, grid, (true,)).unwrap();
        self
    }

    pub fn set_xlabel(&mut self, label: &str) -> &mut Self {
        meth!(self.ax, set_xlabel, (label,)).unwrap();
        self
    }

    pub fn set_ylabel(&mut self, label: &str) -> &mut Self {
        meth!(self.ax, set_ylabel, (label,)).unwrap();
        self
    }

    pub fn legend(&mut self) -> &mut Self {
        meth!(self.ax, legend, ()).unwrap();
        self
    }
}

enum PlotData<'a, D>
where D: ?Sized {
    XY(&'a D, &'a D),
    Y(&'a D),
}

#[derive(Clone)]
struct PlotOptions<'a> {
    fmt: &'a str,
    animated: bool,
    antialiased: bool,
    label: &'a str,
    linewidth: Option<f64>,
}

impl<'a> PlotOptions<'a> {
    fn new() -> PlotOptions<'static> {
        PlotOptions { fmt: "", animated: false, antialiased: true,
                      label: "", linewidth: None }
    }

    fn kwargs(&'a self, py: Python<'a>) -> &'a PyDict {
        let kwargs = PyDict::new(py);
        if self.animated {
            kwargs.set_item("animated", true).unwrap()
        }
        kwargs.set_item("antialiased", self.antialiased).unwrap();
        if !self.label.is_empty() {
            kwargs.set_item("label", self.label).unwrap()
        }
        if let Some(w) = self.linewidth {
            kwargs.set_item("linewidth", w).unwrap()
        }
        kwargs
    }

    fn plot_xy<D>(&self, py: Python<'_>, numpy: &Numpy, axes: &Axes,
        x: &D, y: &D)
    where D: Data + ?Sized {
        let xn = x.to_numpy(py, numpy);
        let yn = y.to_numpy(py, numpy);
        axes.ax.call_method(py, "plot", (xn, yn, self.fmt),
                            Some(self.kwargs(py))).unwrap();
    }

    fn plot_y<D>(&self, py: Python<'_>, numpy: &Numpy, axes: &Axes, y: &D)
    where D: Data + ?Sized {
        let yn = y.to_numpy(py, numpy);
        axes.ax.call_method(py, "plot", (yn, self.fmt),
                            Some(self.kwargs(py))).unwrap();
    }

    fn plot_data<D>(&self, py: Python<'_>, numpy: &Numpy, axes: &Axes,
        data: &PlotData<'_, D>)
    where D: Data + ?Sized {
        match data {
            PlotData::XY(x, y) => {
                self.plot_xy(py, numpy, axes, *x, *y) }
            PlotData::Y(y) => {
                self.plot_y(py, numpy, axes, *y) }
        }
    }

}

/// Declare methods to set the options assuming `self.options` exists.
macro_rules! set_plotoptions { () => {
    #[must_use]
    pub fn fmt(mut self, fmt: &'a str) -> Self {
        self.options.fmt = fmt;
        self
    }

    #[must_use]
    pub fn animated(mut self) -> Self {
        self.options.animated = true;
        self
    }

    #[must_use]
    pub fn antialiased(mut self, b: bool) -> Self {
        self.options.antialiased = b;
        self
    }

    #[must_use]
    pub fn label(mut self, label: &'a str) -> Self {
        self.options.label = label;
        self
    }

    #[must_use]
    pub fn linewidth(mut self, w: f64) -> Self {
        self.options.linewidth = Some(w);
        self
    }
}}

pub struct XY<'a, D>
where D: ?Sized {
    axes: &'a Axes,
    // Latest data and its setting.
    options: PlotOptions<'a>,
    data: PlotData<'a, D>,
    // Previous data with their settings.
    prev_data: Vec<(PlotOptions<'a>, PlotData<'a, D>)>,
}

impl<'a, D> XY<'a, D>
where D: Data + ?Sized {
    set_plotoptions!();

    /// Plot the data with the options specified in [`XY`].
    pub fn plot(self) {
        let numpy = pymod!(NUMPY).unwrap();
        Python::with_gil(|py| {
            for (opt, data) in self.prev_data.iter() {
                opt.plot_data(py, numpy, self.axes, data)
            }
            self.options.plot_data(py, numpy, self.axes, &self.data)
        })
    }

    /// Add the dataset (`x`, `y`).
    #[must_use]
    pub fn xy(&mut self, x: &'a D, y: &'a D) -> &mut Self {
        let mut data = PlotData::XY(x, y);
        swap(&mut data, &mut self.data);
        self.prev_data.push((self.options.clone(), data));
        self
    }

    /// Add the dataset `y`.
    #[must_use]
    pub fn y(&mut self, y: &'a D) -> &mut Self {
        let mut data = PlotData::Y(y);
        swap(&mut data, &mut self.data);
        self.prev_data.push((self.options.clone(), data));
        self
    }
}

pub struct XYFrom<'a, I> {
    axes: &'a Axes,
    options: PlotOptions<'a>,
    data: I,
}

impl<'a, I> XYFrom<'a, I>
where I: IntoIterator,
      <I as IntoIterator>::Item: Borrow<(f64, f64)> {
    set_plotoptions!();

    /// Plot the data with the options specified in [`XYFrom`].
    pub fn plot(self) {
        let data = self.data.into_iter();
        let n = data.size_hint().0;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for di in data {
            let &(xi, yi) = di.borrow();
            x.push(xi);
            y.push(yi);
        }
        let numpy = pymod!(NUMPY).unwrap();
        Python::with_gil(|py| {
            self.options.plot_xy(py, numpy, self.axes, &x, &y) })
    }
}

/// Options to plot functions (require the library [curve-sampling][]).
///
/// [curve-sampling]: https://crates.io/crates/curve-sampling
#[must_use]
pub struct Fun<'a, F> {
    axes: &'a Axes,
    options: PlotOptions<'a>,
    f: F,
    a: f64, // [a, b] is the interval on which we want to plot f.
    b: f64,
    n: usize,
}

#[cfg(feature = "curve-sampling")]
impl<'a, F> Fun<'a, F>
where F: FnMut(f64) -> f64 {
    set_plotoptions!();

    /// Plot the data with the options specified in [`XY`].
    pub fn plot(mut self) {
        let s = Sampling::fun(&mut self.f, self.a, self.b)
            .n(self.n).build();
        // Ensure `x` and `y` live to the end of the call to "plot".
        let x = s.x();
        let y = s.y();
        let numpy = pymod!(NUMPY).unwrap();
        Python::with_gil(|py| {
            self.options.plot_xy(py, numpy, self.axes, &x, &y) })
    }

    /// Set the maximum number of evaluations of the function to build
    /// the sampling.  Panic if `n` < 2.
    pub fn n(&mut self, n: usize) -> &mut Self {
        if n < 2 {
            panic!("matplotlib::Fun::n: at least two points are required.");
        }
        self.n = n;
        self
    }
}


impl Line2D {
    fn set_kw<'a, I>(&'a self, kwargs: I) -> &'a Self
    where I: IntoPyDict {
        Python::with_gil(|py| {
            let kwargs = Some(kwargs.into_py_dict(py));
            for l in self.line2d.as_ref(py).iter() {
                l.call_method("set", (), kwargs).unwrap();
            }
        });
        self
    }

    pub fn label(&self, label: &str) -> &Self {
        self.set_kw([("label", label)])
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_basic_pdf() -> Result<(), Error> {
        let (fig, [[mut ax]]) = subplots()?;
        dbg!(&fig);
        ax.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
        fig.save().to_file("target/a_basic.pdf")?;
        Ok(())
    }

    #[test]
    fn a_basic_row() -> Result<(), Error> {
        let (fig, [[mut ax0, mut ax1]]) = subplots()?;
        ax0.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
        ax1.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt(".").plot();
        fig.save().to_file("target/a_basic_row.pdf")?;
        Ok(())
    }

    #[test]
    fn a_basic_col() -> Result<(), Error> {
        let (fig, [[mut ax0], [mut ax1]]) = subplots()?;
        ax0.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
        ax1.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt(".").plot();
        fig.save().to_file("target/a_basic_col.pdf")?;
        Ok(())
    }

    #[test]
    fn a_basic_grid() -> Result<(), Error> {
        let (fig, [[mut ax0, mut ax1],
                   [mut ax2, mut ax3]]) = subplots()?;
        ax0.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
        ax1.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt(".").plot();
        ax2.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt("r").plot();
        ax3.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt("r.").plot();
        fig.save().to_file("target/a_basic_grid.pdf")?;
        Ok(())
    }

}
