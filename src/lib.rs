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
    path::Path, pin::Pin, borrow::Cow,
};
use lazy_static::lazy_static;
use pyo3::{
    prelude::*,
    intern,
    exceptions::{PyFileNotFoundError, PyPermissionError},
    types::{PyDict, PyList},
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

impl Numpy {
    /// Convert `vec` to a numpy.ndarray (without copying).
    /// SAFETY: The result depends on `x` to exist as they share data.
    fn ndarray(&self, py: Python, x: Pin<&[f64]>) -> PyObject {
        // ty = ctypes.POINTER(ctypes.c_double)
        let ty = getattr!(py, self.ctypes, "POINTER")
            .call1(py, (getattr!(py, self.ctypes, "c_double"),)).unwrap();
        // ptr = ctypes.cast(x.as_ptr(), ty)
        let ptr = getattr!(py, self.ctypes, "cast")
            .call1(py, (x.as_ptr() as usize, ty)).unwrap();
        // numpy.ctypeslib.as_array(ptr, shape=(x.len(),))
        getattr!(py, self.numpy, "as_array")
            .call1(py, (ptr, (x.len(),))).unwrap()
    }
}

/// Container for most of the (sub-)plot elements: Axis, Tick,
/// [`Line2D`], Text, Polygon, etc., and sets the coordinate system.
#[derive(Debug, Clone)]
pub struct Axes {
    ax: PyObject,
}

/// The top level container for all the plot elements.
#[derive(Debug)]
pub struct Figure {
    fig: PyObject, // instance of matplotlib.figure.Figure
}

/// A line — the line can have both a solid linestyle connecting all
/// the vertices, and a marker at each vertex. Additionally, the
/// drawing of the solid line is influenced by the drawstyle, e.g.,
/// one can create "stepped" lines in various styles.
pub struct Line2D {
    line2d: PyObject,
}

#[inline(always)]
fn grid<const R: usize, const C: usize, U>(
    f: impl Fn(usize, usize) -> U) -> [[U; C]; R]
{
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
    ///
    /// ⚠ The figures created with this function will not be displayed
    /// with [`show`].  They can be [saved][Figure::save] to files.
    pub fn new() -> Result<Figure, Error> {
        let figure = pymod!(FIGURE)?;
        Python::with_gil(|py| {
            let fig = getattr!(py, figure, "Figure")
                .call0(py).unwrap();
            Ok(Self { fig: fig.into() })
        })
    }

    /// Return a grid of subplots with `R` rows and `C` columns.
    pub fn subplots<const R: usize, const C: usize>(
        &self
    ) -> Result<[[Axes; C]; R], Error> {
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
    /// [`show()`] for that.
    ///
    /// [GUI]: https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.show
    pub fn show(self) -> Result<(), Error> {
        Python::with_gil(|py|
            match self.fig.call_method0(py, intern!(py, "show")) {
                Ok(_) => Ok(()),
                Err(e) => Err(Error::Python(e)),
            })
    }

    /// Save the figure to a file.
    pub fn save(&self) -> Savefig {
        Savefig { fig: self.fig.clone(), dpi: None }
    }
}

/// Options for saving figures.
#[must_use]
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


/// Return a new figure.
/// This figure is tracked by Matplotlib so [`show()`] displays it.
/// This implies it must be explicitly deallocated using [`close`].
pub fn figure() -> Result<Figure, Error> {
    let pyplot = pymod!(PYPLOT)?;
    Python::with_gil(|py| {
        let fig = getattr!(py, pyplot, "figure")
            .call0(py).map_err(|e| Error::Python(e))?;
        Ok(Figure { fig: fig.into() })
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
    let pyplot = pymod!(PYPLOT).unwrap();
    Python::with_gil(|py| {
        // FIXME: What do we want to do with the errors?
        getattr!(py, pyplot, "show").call0(py).unwrap();
    })
}

/// Close the figure `fig` (created with [`figure`] or [`subplots`]).
pub fn close(fig: Figure) {
    let pyplot = pymod!(PYPLOT).unwrap();
    Python::with_gil(|py| {
        getattr!(py, pyplot, "close").call1(py, (fig.fig,)).unwrap();
    })
}

/// Close all figures created with [`figure`] or [`subplots`].
pub fn close_all() {
    let pyplot = pymod!(PYPLOT).unwrap();
    Python::with_gil(|py| {
        getattr!(py, pyplot, "close").call1(py, ("all",)).unwrap();
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
    pub fn xy<'a, D>(&'a mut self, x: D, y: D) -> XY<'a, D>
    where D: AsRef<[f64]> {
        // The chain leading to plot starts with the data (using this
        // function) so that additional data may be added, sharing
        // common options.  We also mutably borrow `self` to reflect that
        // the final `.plot()` will mutate the underlying Python object.
        XY { axes: self,
             options: PlotOptions::new(),
             data: PlotData::XY(x, y),
        }
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
    pub fn y<'a, D>(&'a mut self, y: D) -> XY<'a, D>
    where D: AsRef<[f64]> {
        XY { axes: self,
             options: PlotOptions::new(),
             data: PlotData::Y(y),
        }
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
    pub fn xy_from<'a, I>(&'a mut self, xy: I) -> XYFrom<'a, I>
    where I: IntoIterator, <I as IntoIterator>::Item: CoordXY {
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
    pub fn scatter<D>(&mut self, x: D, y: D) -> &mut Self
    where D: AsRef<[f64]> {
        // FIXME: Do we want to check that `x` and `y` have the same
        // dimension?  Better error message?
        let numpy = pymod!(NUMPY).unwrap();
        let x = Pin::new(x.as_ref());
        let y = Pin::new(y.as_ref());
        meth!(self.ax, scatter, py -> {
            let xn = numpy.ndarray(py, x);
            let yn = numpy.ndarray(py, y);
            (xn, yn) })
            .unwrap();
        self
    }

    /// Set the title to `txt` for the Axes.
    pub fn set_title(&mut self, txt: impl AsRef<str>) -> &mut Self {
        meth!(self.ax, set_title, (txt.as_ref(),)).unwrap();
        self
    }

    /// Set the yaxis' scale.  Possible values for `v` are "linear",
    /// "log", "symlog", "logit",...
    pub fn set_yscale(&mut self, v: &str) -> &mut Self {
        meth!(self.ax, set_yscale, (v,)).unwrap();
        self
    }

    /// Configure the grid lines.
    pub fn grid(&mut self) -> &mut Self {
        meth!(self.ax, grid, (true,)).unwrap();
        self
    }

    /// Set the label for the X-axis.
    pub fn set_xlabel(&mut self, label: impl AsRef<str>) -> &mut Self {
        meth!(self.ax, set_xlabel, (label.as_ref(),)).unwrap();
        self
    }

    /// Set the label for the Y-axis.
    pub fn set_ylabel(&mut self, label: impl AsRef<str>) -> &mut Self {
        meth!(self.ax, set_ylabel, (label.as_ref(),)).unwrap();
        self
    }

    /// Place a legend on the Axes whose elements are taken from
    /// `lines`.  If `lines` is empty, the elements are automatically
    /// determined from the labels specified to the axis plots.
    pub fn legend<L, U>(&mut self, lines: L) -> &mut Self
    where L: IntoIterator<Item=Line2D, IntoIter = U>,
          U: ExactSizeIterator<Item = Line2D>
    {
        Python::with_gil(|py| {
            let elements = lines.into_iter().map(|l| l.line2d);
            let kwargs;
            if elements.len() == 0 { // FIXME: .is_empty is unstable
                kwargs = None;
            } else {
                let dic = PyDict::new(py);
                dic.set_item("handles", PyList::new(py, elements)).unwrap();
                kwargs = Some(dic)
            }
            self.ax.call_method(py, intern!(py, "legend"), (), kwargs)
                .unwrap();
            self
        })
    }

    pub fn twinx(&mut self) -> Axes {
        Axes { ax: meth!(self.ax, twinx, ()).unwrap() }
    }
}

enum PlotData<D> {
    XY(D, D),
    Y(D),
}

#[derive(Clone)]
struct PlotOptions<'a> {
    fmt: &'a str,
    animated: bool,
    antialiased: bool,
    label: Cow<'a, str>,
    linewidth: Option<f64>,
}

impl<'a> PlotOptions<'a> {
    fn new() -> PlotOptions<'static> {
        PlotOptions {
            fmt: "", animated: false, antialiased: true,
            label: Cow::Borrowed(""), linewidth: None
        }
    }

    fn kwargs(&'a self, py: Python<'a>) -> &'a PyDict {
        let kwargs = PyDict::new(py);
        if self.animated {
            kwargs.set_item("animated", true).unwrap()
        }
        kwargs.set_item("antialiased", self.antialiased).unwrap();
        if !self.label.is_empty() {
            let label: &str = self.label.as_ref();
            kwargs.set_item("label", label).unwrap()
        }
        if let Some(w) = self.linewidth {
            kwargs.set_item("linewidth", w).unwrap()
        }
        kwargs
    }

    /// Plot the ndarrays `x` and `y` and return the corresponding line.
    fn plot_xy(&self, py: Python<'_>, numpy: &Numpy, axes: &Axes,
               x: Pin<&[f64]>, y: Pin<&[f64]>) -> Line2D
    {
        let x = numpy.ndarray(py, x);
        let y = numpy.ndarray(py, y);
        let lines = axes.ax.call_method(py,
            "plot", (x, y, self.fmt), Some(self.kwargs(py))).unwrap();
        let lines: &PyList = lines.downcast(py).unwrap();
        // Extract the element from the list of length 1 (1 data plotted)
        let line2d = lines.get_item(0).unwrap().into();
        Line2D { line2d }
    }

    fn plot_y(&self, py: Python<'_>, numpy: &Numpy, axes: &Axes,
              y: Pin<&[f64]>) -> Line2D
    {
        let y = numpy.ndarray(py, y);
        let lines = axes.ax.call_method(py,
            "plot", (y, self.fmt), Some(self.kwargs(py))).unwrap();
        let lines: &PyList = lines.downcast(py).unwrap();
        let line2d = lines.get_item(0).unwrap().into();
        Line2D { line2d }
    }

    fn plot_data<D>(&self, py: Python<'_>, numpy: &Numpy, axes: &Axes,
                    data: &PlotData<D>) -> Line2D
    where D: AsRef<[f64]> {
        match data {
            PlotData::XY(x, y) => {
                let x = Pin::new(x.as_ref());
                let y = Pin::new(y.as_ref());
                self.plot_xy(py, numpy, axes, x, y) }
            PlotData::Y(y) => {
                let y = Pin::new(y.as_ref());
                self.plot_y(py, numpy, axes, y) }
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

    /// Label the plot with `label`.  Note that labels are not shown
    /// by default; one must call [`Axes::legend`] to display them.
    #[must_use]
    pub fn label(mut self, label: impl Into<Cow<'a, str>>) -> Self {
        self.options.label = label.into();
        self
    }

    #[must_use]
    pub fn linewidth(mut self, w: f64) -> Self {
        self.options.linewidth = Some(w);
        self
    }
}}

/// Options to plot X-Y data.  Created by [`Axes::xy`] and [`Axes::y`].
#[must_use]
pub struct XY<'a, D> {
    axes: &'a Axes,
    options: PlotOptions<'a>,
    data: PlotData<D>,
}

impl<'a, D> XY<'a, D>
where D: AsRef<[f64]> {
    set_plotoptions!();

    /// Plot the data with the options specified in [`XY`].
    pub fn plot(self) -> Line2D {
        let numpy = pymod!(NUMPY).unwrap();
        Python::with_gil(|py| {
            self.options.plot_data(py, numpy, self.axes, &self.data)
        })
    }
}

/// Options to plot X-Y data.  Created by [`Axes::xy_from`].
#[must_use]
pub struct XYFrom<'a, I> {
    axes: &'a Axes,
    options: PlotOptions<'a>,
    data: I,
}

pub trait CoordXY {
    fn x(&self) -> f64;
    fn y(&self) -> f64;
}

impl<T> CoordXY for &T where T: CoordXY {
    #[inline]
    fn x(&self) -> f64 { (*self).x() }
    #[inline]
    fn y(&self) -> f64 { (*self).y() }
}

impl CoordXY for (f64, f64) {
    #[inline]
    fn x(&self) -> f64 { self.0 }
    #[inline]
    fn y(&self) -> f64 { self.1 }
}

impl CoordXY for [f64; 2] {
    #[inline]
    fn x(&self) -> f64 { self[0] }
    #[inline]
    fn y(&self) -> f64 { self[1] }
}

impl<'a, I> XYFrom<'a, I>
where I: IntoIterator,
      <I as IntoIterator>::Item: CoordXY {
    set_plotoptions!();

    /// Plot the data with the options specified in [`XYFrom`].
    pub fn plot(self) -> Line2D {
        let data = self.data.into_iter();
        let n = data.size_hint().0;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for di in data {
            x.push(di.x());
            y.push(di.y());
        }
        let x = Pin::new(x.as_slice());
        let y = Pin::new(y.as_slice());
        let numpy = pymod!(NUMPY).unwrap();
        Python::with_gil(|py| {
            self.options.plot_xy(py, numpy, self.axes, x, y) })
    }
}

/// Options to plot functions (require the library [curve-sampling][]).
/// Created by [`Axes::fun`].
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
    pub fn plot(mut self) -> Line2D {
        let s = Sampling::fun(&mut self.f, self.a, self.b)
            .n(self.n).build();
        // Ensure `x` and `y` live to the end of the call to "plot".
        let x = s.x();
        let y = s.y();
        let x = Pin::new(x.as_slice());
        let y = Pin::new(y.as_slice());
        let numpy = pymod!(NUMPY).unwrap();
        Python::with_gil(|py| {
            self.options.plot_xy(py, numpy, self.axes, x, y)
        })
    }

    /// Set the maximum number of evaluations of the function to build
    /// the sampling.  Panic if `n` < 2.
    pub fn n(mut self, n: usize) -> Self {
        if n < 2 {
            panic!("matplotlib::Fun::n: at least two points are required.");
        }
        self.n = n;
        self
    }
}


impl Line2D {
    fn set_kw(&self, prop: &str, v: impl ToPyObject) {
        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item(prop, v).unwrap();
            self.line2d.call_method(py, "set", (), Some(kwargs)).unwrap();
        })
    }

    pub fn set_label(&mut self, label: impl AsRef<str>) -> &mut Self {
        self.set_kw("label", label.as_ref());
        self
    }

    pub fn set_color(&mut self, c: f64) -> &mut Self {
        meth!(self.line2d, set_color, (c,)).unwrap();
        self
    }

    pub fn set_linewidth(&mut self, w: f64) -> &mut Self {
        self.set_kw("linewidth", w);
        self
    }

    pub fn linewidth(self, w: f64) -> Self {
        self.set_kw("linewidth", w);
        self
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
    fn a_basic_label() -> Result<(), Error> {
        let (fig, [[mut ax]]) = subplots()?;
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

    #[test]
    //#[compile_fail]
    fn data_in_scope() -> Result<(), Error> {
        let (fig, [[mut ax]]) = subplots()?;
        // let l = ax.y(&vec![1., 2.]);
        // l.plot();
        ax.y(&vec![1., 2.]).plot();
        fig.save().to_file("target/data_in_scope.pdf")?;
        Ok(())
    }

}
