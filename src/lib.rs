use std::{path::Path,
          fmt::{Display, Formatter}, mem::swap};

use pyo3::{prelude::*,
           exceptions::{PyFileNotFoundError, PyPermissionError},
           types::{PyTuple, IntoPyDict, PyDict, PyList}, intern};
use numpy::{PyArray1, PyArray2};

#[cfg(feature = "curve-sampling")]
use curve_sampling::Sampling;

macro_rules! py  {
    ($py: ident, $lib: expr, $f: ident) => {
        $lib.getattr($py, intern!($py, stringify!($f))).unwrap()
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
    ax: PyObject,
    numpy: Numpy,
}

impl Axes {
    fn new(py: Python, ax: PyObject) -> Self {
        Self { ax,  numpy: Numpy::new(py).unwrap() }
    }
}


pub struct Figure {
    plt: Py<PyModule>, // module matplotlib.pyplot
    figure: PyObject,
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
    /// Plot `y` versus `x` as lines and/or markers.
    ///
    /// # Example
    ///
    /// ```
    /// use matplotlib::Plot;
    /// let (fig, [[mut ax]]) = Plot::sub()?;
    /// ax.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
    /// fig.savefig("target/XY_plot.pdf")?;
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
    /// use matplotlib::Plot;
    /// let (fig, [[mut ax]]) = Plot::sub()?;
    /// ax.y(&[1., 4., 2., 3.]).plot();
    /// fig.savefig("target/Y_plot.pdf")?;
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

    #[must_use]
    pub fn xy_from<'a, I>(&'a mut self, xy: I) -> XYFrom<'a, I>
    where I: IntoIterator<Item = (f64, f64)> {
        XYFrom { axes: self,
                 options: PlotOptions::new(),
                 data: xy }
    }

    #[cfg(feature = "curve-sampling")]
    /// Plot the graph of the function `f` on the interval \[`a`, `b`\].
    ///
    /// # Example
    /// ```
    /// use matplotlib::Plot;
    /// let (fig, [[mut ax]]) = Plot::sub()?;
    /// ax.fun(|x| x * x, 0., 1.).plot();
    /// fig.savefig("target/Fun_plot.pdf")?;
    /// # Ok::<(), matplotlib::Error>(())
    /// ```
    pub fn fun<'a, F>(&'a mut self, f: F, a: f64, b: f64) -> Fun<'a, F>
    where F: FnMut(f64) -> f64 {
        Fun { axes: self,
              options: PlotOptions::new(),
              f, a, b,
              n: 100 }
    }


    pub fn scatter<D>(&mut self, x: &D, y: &D) -> &mut Self
    where D: Data + ?Sized {
        // FIXME: Do we want to check that `x` and `y` have the same
        // dimension?  Better error message?
        meth!(self.ax, scatter, py -> {
            let xn = x.to_numpy(py, &self.numpy);
            let yn = y.to_numpy(py, &self.numpy);
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

    pub fn legend(&mut self) {
        meth!(self.ax, legend, ()).unwrap();
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

    fn plot_xy<D>(&self, py: Python<'_>, axes: &Axes, x: &D, y: &D)
    where D: Data + ?Sized {
        let xn = x.to_numpy(py, &axes.numpy);
        let yn = y.to_numpy(py, &axes.numpy);
        axes.ax.call_method(py, "plot", (xn, yn, self.fmt),
                            Some(self.kwargs(py))).unwrap();
    }

    fn plot_y<D>(&self, py: Python<'_>, axes: &Axes, y: &D)
    where D: Data + ?Sized {
        let yn = y.to_numpy(py, &axes.numpy);
        axes.ax.call_method(py, "plot", (yn, self.fmt),
                            Some(self.kwargs(py))).unwrap();
    }

    fn plot_data<D>(&self, py: Python<'_>, axes: &Axes, data: &PlotData<'_, D>)
    where D: Data + ?Sized {
        match data {
            PlotData::XY(x, y) => self.plot_xy(py, axes, *x, *y),
            PlotData::Y(y) => self.plot_y(py, axes, *y),
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
        Python::with_gil(|py| {
            for (opt, data) in self.prev_data.iter() {
                opt.plot_data(py, self.axes, data)
            }
            self.options.plot_data(py, self.axes, &self.data)
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
where I: IntoIterator<Item = (f64, f64)> {
    set_plotoptions!();

    /// Plot the data with the options specified in [`XYFrom`].
    pub fn plot(self) {
        let data = self.data.into_iter();
        let n = data.size_hint().0;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for (xi, yi) in data {
            x.push(xi);
            y.push(yi);
        }
        Python::with_gil(|py| self.options.plot_xy(py, self.axes, &x, &y))
    }
}

/// Options to plot functions (require the library [curve-sampling][]).
///
/// [curve-sampling]: https://crates.io/crates/curve-sampling
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
        Python::with_gil(|py| self.options.plot_xy(py, self.axes, &x, &y))
    }

    /// Set the maximum number of evaluations of the function to build
    /// the sampling.  Panic if n < 2.
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

    fn label(&self, label: &str) -> &Self {
        self.set_kw([("label", label)])
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
        meth!(self.figure, savefig, py -> (path.as_ref(),), e -> {
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
        ax.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
        fig.savefig("target/a_basic.pdf")?;
        Ok(())
    }

    #[test]
    fn a_basic_row() -> Result<(), Error> {
        let (fig, [[mut ax0, mut ax1]]) = Plot::sub()?;
        ax0.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
        ax1.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt(".").plot();
        fig.savefig("target/a_basic_row.pdf")?;
        Ok(())
    }

    #[test]
    fn a_basic_col() -> Result<(), Error> {
        let (fig, [[mut ax0], [mut ax1]]) = Plot::sub()?;
        ax0.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
        ax1.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt(".").plot();
        fig.savefig("target/a_basic_col.pdf")?;
        Ok(())
    }

    #[test]
    fn a_basic_grid() -> Result<(), Error> {
        let (fig, [[mut ax0, mut ax1],
                   [mut ax2, mut ax3]]) = Plot::sub()?;
        ax0.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).plot();
        ax1.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt(".").plot();
        ax2.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt("r").plot();
        ax3.xy(&[1., 2., 3., 4.], &[1., 4., 2., 3.]).fmt("r.").plot();
        fig.savefig("target/a_basic_grid.pdf")?;
        Ok(())
    }

}
