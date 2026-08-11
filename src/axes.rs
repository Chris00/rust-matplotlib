//! [`Axes`] represents one (sub-)plot in a figure.
//!
//! It contains the plotted data, axis ticks, labels, title, legend,
//! etc.  Its methods are the main interface for manipulating the
//! plot.

use crate::{
    colors::{self, Color},
    lines::Line2D,
    text::Text,
};
use ndarray::{Array1, Array2};
use numpy::{Element, PyArray1, convert::ToPyArray};
use pyo3::{
    intern,
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use std::marker::PhantomData;

#[cfg(feature = "curve-sampling")]
use curve_sampling::Sampling;

include!("macros.rs");

/// Container for most of the (sub-)plot elements: Axis, Tick,
/// [`Line2D`], Text, Polygon, etc., and sets the coordinate system.
#[derive(Debug)]
pub struct Axes {
    pub(crate) ax: Py<PyAny>,
}

/// Alias for one-dimensional bound-arrays in Python.
pub type PyVector<'py, T> = Bound<'py, PyArray1<T>>;

/// Types convertible to one-dimensional Python arrays.
pub trait Vector<T: Element> {
    fn to_pyvector<'py>(&self, py: Python<'py>) -> PyVector<'py, T>;
}

impl<T: Element> Vector<T> for [T] {
    fn to_pyvector<'py>(&self, py: Python<'py>) -> PyVector<'py, T> {
        PyArray1::from_slice(py, self)
    }
}

impl<T: Element> Vector<T> for Array1<T> {
    fn to_pyvector<'py>(&self, py: Python<'py>) -> PyVector<'py, T> {
        <Self as ToPyArray>::to_pyarray(self, py)
    }
}

#[cfg(feature = "nalgebra")]
impl<T, R, S> Vector<T> for nalgebra::Vector<T, R, S>
where
    T: nalgebra::Scalar + Element,
    R: nalgebra::Dim,
    S: nalgebra::Storage<T, R, nalgebra::base::dimension::U1>,
{
    fn to_pyvector<'py>(&self, py: Python<'py>) -> PyVector<'py, T> {
        // Based on `numpy::convert::ToPyArray` impl for `nalgebra::Matrix`
        // but returning a one-dimensional array.
        use numpy::PyArrayMethods;
        unsafe {
            let vec = PyArray1::<T>::new(py, (self.len(),), true);
            let mut data_ptr = vec.data();
            if self.data.is_contiguous() {
                std::ptr::copy_nonoverlapping(self.data.ptr(), data_ptr, self.len());
            } else {
                for item in self.iter() {
                    data_ptr.write(item.clone_ref(py));
                    data_ptr = data_ptr.add(1);
                }
            }
            vec
        }
    }
}

impl<T: Element> Vector<T> for Vec<T> {
    fn to_pyvector<'py>(&self, py: Python<'py>) -> PyVector<'py, T> {
        PyArray1::from_slice(py, self)
    }
}

impl<T: Element, const N: usize> Vector<T> for [T; N] {
    fn to_pyvector<'py>(&self, py: Python<'py>) -> PyVector<'py, T> {
        PyArray1::from_slice(py, self)
    }
}

/// An Axes struct encapsulates all the elements of an individual
/// (sub-)plot in a figure.
impl Axes {
    /// Plot `y` versus `x` as lines and/or markers.
    ///
    /// # Example
    ///
    /// ```
    /// use matplotlib::{pyplot as plt, colors};
    /// let (fig, [[mut ax]]) = plt::subplots()?;
    /// let x = [1., 2., 3., 4.];
    /// let y = [1., 4., 2., 3.];
    /// ax.xy(&x, &y).fmt("-").color(colors::Base::R).plot();
    /// ax.xy(&x, &y).fmt("bo").plot();
    /// fig.save().to_file("target/XY_plot.pdf")?;
    /// # Ok::<(), matplotlib::Error>(())
    /// ```
    // FIXME: Do we want to check that `x` and `y` have the same
    // dimension?  Better error message?
    pub fn xy<'a>(
        &'a mut self,
        x: &'a (impl Vector<f64> + ?Sized),
        y: &'a (impl Vector<f64> + ?Sized),
    ) -> XY<'a> {
        // The chain leading to plot starts with the data (using this
        // function) so that additional data may be added, sharing
        // common options.  We also mutably borrow `self` to reflect that
        // the final `.plot()` will mutate the underlying Python object.
        XY::xy(self, x, y)
    }

    /// Plot `y` versus its indices as lines and/or markers.
    ///
    /// # Example
    ///
    /// ```
    /// use matplotlib::pyplot as plt;
    /// let (fig, [[mut ax]]) = plt::subplots()?;
    /// ax.y(&[1., 4., 2., 3.]).plot();
    /// fig.save().to_file("target/Y_plot.pdf")?;
    /// # Ok::<(), matplotlib::Error>(())
    /// ```
    pub fn y<'a>(&'a mut self, y: &'a (impl Vector<f64> + ?Sized)) -> XY<'a> {
        XY::y(self, y)
    }

    /// Convenience function to plot X-Y coordinates coming from `xy`.
    ///
    /// # Example
    ///
    /// ```
    /// use matplotlib::pyplot as plt;
    /// let (fig, [[mut ax]]) = plt::subplots()?;
    /// ax.xy_from(&[(1., 2.), (4., 2.), (2., 3.), (3., 4.)]).plot();
    /// ax.xy_from((1..=4).map(f64::from).zip([0., 3., 1., 3.])).plot();
    /// fig.save().to_file("target/XY_from_plot.pdf")?;
    /// # Ok::<(), matplotlib::Error>(())
    /// ```
    pub fn xy_from<'a, I>(&'a mut self, xy: I) -> XYFrom<'a, I>
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: CoordXY,
    {
        XYFrom::new(self, xy)
    }

    #[cfg(feature = "curve-sampling")]
    /// Plot the graph of the function `f` on the interval \[`a`, `b`\].
    ///
    /// # Example
    /// ```
    /// use matplotlib::pyplot as plt;
    /// let (fig, [[mut ax]]) = plt::subplots()?;
    /// ax.fun(|x| x * x, 0., 1.).plot();
    /// fig.save().to_file("target/Fun_plot.pdf")?;
    /// # Ok::<(), matplotlib::Error>(())
    /// ```
    pub fn fun<'a, F, Y, D>(&'a mut self, f: F, a: f64, b: f64) -> Fun<'a, F, D>
    where
        F: FnMut(f64) -> Y,
        Y: curve_sampling::Img<D>,
    {
        Fun::new(self, f, a, b)
    }

    /// Draw the contour lines for the data `z[j,i]` as points
    /// (`x[i]`, `y[j]`).
    ///
    /// # Example
    ///
    /// ```
    /// use matplotlib::{pyplot as plt, colors::Tab};
    /// use ndarray::{Array1, Array2};
    /// let x: Array1<f64> = Array1::linspace(-1., 1., 30);
    /// let y: Array1<f64> = Array1::linspace(-1., 1., 30);
    /// let mut z = Array2::zeros((30, 30));
    /// for (j, &y) in y.iter().enumerate() {
    ///     for (i, &x) in x.iter().enumerate() {
    ///         z[(j, i)] = (0.5 * x).powi(2) + y.powi(2);
    ///     }
    /// }
    /// let (fig, [[mut ax]]) = plt::subplots()?;
    /// ax.contour(&x, &y, &z)
    ///     .levels(&[0.2, 0.5, 0.8])
    ///     .colors(&[Tab::Red, Tab::Blue, Tab::Olive])
    ///     .plot();
    /// fig.save().to_file("target/contour.pdf")?;
    /// # Ok::<(), matplotlib::Error>(())
    /// ```
    pub fn contour<'a>(
        &'a mut self,
        x: &'a (impl Vector<f64> + ?Sized),
        y: &'a (impl Vector<f64> + ?Sized),
        z: &'a ndarray::Array2<f64>,
    ) -> Contour<'a> {
        Contour::new(self, x, y, z)
    }

    /// Draw the contour lines for function `f` in the rectangle `ab`×`cd`.
    ///
    /// # Example
    ///
    /// ```
    /// use matplotlib::pyplot as plt;
    /// let (fig, [[mut ax]]) = plt::subplots()?;
    /// ax.contour_fun([-1., 1.], [-1., 1.], |x, y| {
    ///     (0.5 * x).powi(2) + y.powi(2)
    /// })
    ///     .plot();
    /// fig.save().to_file("target/contour_fun.pdf")?;
    /// # Ok::<(), matplotlib::Error>(())
    /// ```
    pub fn contour_fun<'a, F>(
        &'a mut self,
        ab: [f64; 2],
        cd: [f64; 2],
        f: F,
    ) -> ContourFun<'a, F>
    where
        F: FnMut(f64, f64) -> f64,
    {
        ContourFun::new(self, ab, cd, f)
    }

    /// Scatter plot of `y` vs. `x` with optional varying marker size
    /// and/or color.
    pub fn scatter<'a>(
        &'a mut self,
        x: &'a (impl Vector<f64> + ?Sized),
        height: &'a (impl Vector<f64> + ?Sized),
    ) -> Scatter<'a> {
        Scatter::new(self, x, height)
    }

    /// Make a bar plot.
    ///
    /// The bars are positioned at `x` with the given
    /// [alignment][`Bar::align`].  Their dimensions are given by
    /// height and width. The vertical baseline is bottom (default 0).
    pub fn bar<'a>(
        &'a mut self,
        x: &'a (impl Vector<f64> + ?Sized),
        height: &'a (impl Vector<f64> + ?Sized),
    ) -> Bar<'a> {
        Bar::new(self, x, height)
    }

    /// Create a stem plot.
    ///
    /// A stem plot draws lines perpendicular to a baseline at each
    /// location locs from the baseline to heads, and places a marker
    /// there.  For vertical stem plots (the default), the locs are
    /// `x` positions, and the heads are `y` values.  For horizontal
    /// stem plots, the locs are `y` positions, and the heads are `x`
    /// values.
    pub fn stem<'a>(
        &'a mut self,
        x: &'a (impl Vector<f64> + ?Sized),
        y: &'a (impl Vector<f64> + ?Sized),
    ) -> Stem<'a> {
        Stem::new(self, x, y)
    }

    pub fn fill_between<'a>(
        &'a mut self,
        x: &'a (impl Vector<f64> + ?Sized),
        y1: &'a (impl Vector<f64> + ?Sized),
        y2: &'a (impl Vector<f64> + ?Sized),
    ) -> FillBetween<'a> {
        FillBetween::new(self, x, y1, y2)
    }

    pub fn stack<'a>(
        &'a mut self,
        x: &'a (impl Vector<f64> + ?Sized),
        y: &'a ndarray::Array2<f64>,
    ) -> Stack<'a> {
        Stack::new(self, x, y)
    }

    /// Draw a stepwise constant function as a line or a filled plot.
    ///
    /// `values` is the function values between these steps.
    /// Depending on [`fill`][Stairs::fill], the function is drawn
    /// either as a continuous line with vertical segments at the
    /// edges, or as a filled area.
    pub fn stairs<'a>(
        &'a mut self,
        values: &'a (impl Vector<f64> + ?Sized),
    ) -> Stairs<'a> {
        Stairs::new(self, values)
    }

    /// Set the title to `txt` for the Axes.
    pub fn set_title(&mut self, txt: impl AsRef<str>) -> &mut Self {
        meth!(self.ax, set_title, (txt.as_ref(),)).unwrap();
        self
    }
    
    /// Set the xaxis' scale.  Possible values for `v` are "linear",
    /// "log", "symlog", "logit",...
    pub fn set_xscale(&mut self, v: &str) -> &mut Self {
        meth!(self.ax, set_xscale, (v,)).unwrap();
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

    /// Set the X-axis view limits.
    pub fn set_xlim(&mut self, x_min: f64, x_max: f64) -> &mut Self {
        let left = if x_min.is_finite() { Some(x_min) } else { None };
        let right = if x_max.is_finite() { Some(x_max) } else { None };
        meth!(self.ax, set_xlim, (left, right)).unwrap();
        self
    }

    /// Set the Y-axis view limits.
    pub fn set_ylim(&mut self, y_min: f64, y_max: f64) -> &mut Self {
        let bottom = if y_min.is_finite() { Some(y_min) } else { None };
        let top = if y_max.is_finite() { Some(y_max) } else { None };
        meth!(self.ax, set_ylim, (bottom, top)).unwrap();
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

    /// Set the xaxis' tick locations and optionally tick labels.
    // FIXME: ticks labels
    pub fn set_xticks(&mut self, ticks: impl IntoIterator<Item=f64>) -> &mut Self {
        let ticks: Vec<_> = ticks.into_iter().collect();
        meth!(self.ax, set_xticks, (ticks,)).unwrap();
        self
    }

    /// Set the yaxis' tick locations and optionally tick labels.
    pub fn set_yticks(&mut self, ticks: impl IntoIterator<Item=f64>) -> &mut Self {
        let ticks: Vec<_> = ticks.into_iter().collect();
        meth!(self.ax, set_yticks, (ticks,)).unwrap();
        self
    }

    /// Place a legend on the Axes whose elements are taken from
    /// `lines`.  If `lines` is empty, the elements are automatically
    /// determined from the labels specified to the axis plots.
    pub fn legend<L, U>(&mut self, lines: L) -> &mut Self
    where
        L: IntoIterator<Item = Line2D, IntoIter = U>,
        U: ExactSizeIterator<Item = Line2D>,
    {
        Python::attach(|py| {
            let elements = lines.into_iter().map(|l| l.line2d);
            if elements.len() == 0 {
                // FIXME: .is_empty is unstable
                self.ax
                    .call_method(py, intern!(py, "legend"), (), None)
                    .unwrap();
            } else {
                let dic = PyDict::new(py);
                dic.set_item("handles", PyList::new(py, elements).unwrap())
                    .unwrap();
                self.ax
                    .call_method(py, intern!(py, "legend"), (), Some(&dic))
                    .unwrap();
            }
            self
        })
    }

    pub fn twinx(&mut self) -> Self {
        Axes {
            ax: meth!(self.ax, twinx, ()).unwrap(),
        }
    }

    pub fn xaxis_date(&mut self) {
        meth!(self.ax, xaxis_date, ()).unwrap();
    }

    pub fn get_xticklabels(&mut self) -> Vec<Text> {
        Python::attach(|py| {
            let labels: Vec<_> = self
                .ax
                .bind(py)
                .call_method1(intern!(py, "get_xticklabels"), ())
                .unwrap()
                .extract()
                .unwrap();
            labels
        })
    }

    /// Display minor ticks on the Axes.
    ///
    /// Displaying minor ticks may reduce performance; you may turn
    /// them off using [`Axes::minorticks_off()`] if drawing speed is
    /// a problem.
    pub fn minorticks_on(&mut self) -> &mut Self {
        meth!(self.ax, minorticks_on, ()).unwrap();
        self
    }

    /// Remove minor ticks from the Axes.
    ///
    /// See also [`Axes::minorticks_on`].
    pub fn minorticks_off(&mut self) -> &mut Self {
        meth!(self.ax, minorticks_off, ()).unwrap();
        self
    }
}

enum PlotData {
    XY(Py<PyArray1<f64>>, Py<PyArray1<f64>>),
    Y(Py<PyArray1<f64>>),
}

#[derive(Clone)]
struct PlotOptions<'a> {
    fmt: &'a str,
    animated: bool,
    antialiased: bool,
    color: Option<[f64; 4]>, // RGBA, if specified
    label: &'a str,
    linewidth: Option<f64>,
    markeredgewidth: Option<f64>,
    markersize: Option<f64>,
    scalex: bool,
    scaley: bool,
}

impl<'a> PlotOptions<'a> {
    fn new() -> PlotOptions<'static> {
        PlotOptions {
            fmt: "",
            animated: false,
            antialiased: true,
            color: None,
            label: "",
            linewidth: None,
            markeredgewidth: None,
            markersize: None,
            scalex: true, // Default
            scaley: true, // Default
        }
    }

    fn kwargs(&'a self, py: Python<'a>) -> Bound<'a, PyDict> {
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
        if let Some(w) = self.markeredgewidth {
            kwargs.set_item("markeredgewidth", w).unwrap()
        }
        if let Some(w) = self.markersize {
            kwargs.set_item("markersize", w).unwrap()
        }
        if let Some(rgba) = self.color {
            let color = PyTuple::new(py, rgba).unwrap();
            kwargs.set_item("color", color).unwrap()
        }
        kwargs.set_item("scalex", self.scalex).unwrap();
        kwargs.set_item("scaley", self.scaley).unwrap();
        kwargs
    }

    /// Plot the ndarrays `x` and `y` and return the corresponding line.
    fn plot_xy(
        &self,
        py: Python,
        axes: &Axes,
        x: &PyVector<f64>,
        y: &PyVector<f64>,
    ) -> Line2D {
        let lines = axes
            .ax
            .call_method(py, "plot", (x, y, self.fmt), Some(&self.kwargs(py)))
            .unwrap();
        let lines: &Bound<PyList> = lines.cast_bound(py).unwrap();
        // Extract the element from the list of length 1 (1 data plotted)
        let line2d = lines.get_item(0).unwrap().into();
        Line2D { line2d }
    }

    fn plot_y(&self, py: Python, axes: &Axes, y: &PyVector<f64>) -> Line2D {
        let lines = axes
            .ax
            .call_method(py, "plot", (y, self.fmt), Some(&self.kwargs(py)))
            .unwrap();
        let lines: &Bound<PyList> = lines.cast_bound(py).unwrap();
        let line2d = lines.get_item(0).unwrap().into();
        Line2D { line2d }
    }

    fn plot_data(&self, py: Python, axes: &Axes, data: PlotData) -> Line2D {
        match data {
            PlotData::XY(x, y) => self.plot_xy(py, axes, x.bind(py), y.bind(py)),
            PlotData::Y(y) => self.plot_y(py, axes, y.bind(py)),
        }
    }
}

/// Declare methods to set the options assuming `self.options` exists.
macro_rules! set_plotoptions {
    () => {
        pub fn fmt(mut self, fmt: &'a str) -> Self {
            self.options.fmt = fmt;
            self
        }

        pub fn animated(mut self) -> Self {
            self.options.animated = true;
            self
        }

        pub fn antialiased(mut self, b: bool) -> Self {
            self.options.antialiased = b;
            self
        }

        /// Label the plot with `label`.  Note that labels are not shown
        /// by default; one must call [`Axes::legend`] to display them.
        pub fn label(mut self, label: &'a str) -> Self {
            self.options.label = label;
            self
        }

        pub fn linewidth(mut self, w: f64) -> Self {
            self.options.linewidth = Some(w);
            self
        }

        pub fn markeredgewidth(mut self, w: f64) -> Self {
            self.options.markeredgewidth = Some(w);
            self
        }

        pub fn markersize(mut self, w: f64) -> Self {
            self.options.markersize = Some(w);
            self
        }

        /// Set the color of the plot.
        pub fn color(mut self, color: impl Color) -> Self {
            self.options.color = Some(color.rgba());
            self
        }

        pub fn scalex(mut self, b: bool) -> Self {
            self.options.scalex = b;
            self
        }

        pub fn scaley(mut self, b: bool) -> Self {
            self.options.scaley = b;
            self
        }
    };
}

/// Options to plot X-Y data.  Created by [`Axes::xy`] and [`Axes::y`].
#[must_use]
pub struct XY<'a> {
    axes: &'a Axes,
    data: PlotData,
    options: PlotOptions<'a>,
}

impl<'a> XY<'a> {
    #[allow(clippy::self_named_constructors)]
    fn xy(
        axes: &'a Axes,
        x: &'a (impl Vector<f64> + ?Sized),
        y: &'a (impl Vector<f64> + ?Sized),
    ) -> Self {
        Python::attach(|py| {
            let x = x.to_pyvector(py).unbind();
            let y = y.to_pyvector(py).unbind();
            Self {
                axes,
                options: PlotOptions::new(),
                data: PlotData::XY(x, y),
            }
        })
    }

    fn y(axes: &'a Axes, y: &'a (impl Vector<f64> + ?Sized)) -> Self {
        Python::attach(|py| {
            let y = y.to_pyvector(py).unbind();
            Self {
                axes,
                options: PlotOptions::new(),
                data: PlotData::Y(y),
            }
        })
    }

    set_plotoptions!();

    /// Plot the data with the options specified in [`XY`].
    pub fn plot(self) -> Line2D {
        Python::attach(|py| self.options.plot_data(py, self.axes, self.data))
    }
}

/// Options to plot X-Y data.  Created by [`Axes::xy_from`].
#[must_use]
pub struct XYFrom<'a, I> {
    axes: &'a Axes,
    data: I,
    options: PlotOptions<'a>,
}

/// 2D coordinates of points.
pub trait CoordXY {
    fn x(&self) -> f64;
    fn y(&self) -> f64;
}

impl<T> CoordXY for &T
where
    T: CoordXY,
{
    #[inline]
    fn x(&self) -> f64 {
        (*self).x()
    }
    #[inline]
    fn y(&self) -> f64 {
        (*self).y()
    }
}

impl CoordXY for (f64, f64) {
    #[inline]
    fn x(&self) -> f64 {
        self.0
    }
    #[inline]
    fn y(&self) -> f64 {
        self.1
    }
}

impl CoordXY for (Option<f64>, Option<f64>) {
    #[inline]
    fn x(&self) -> f64 {
        self.0.unwrap_or(f64::NAN)
    }
    #[inline]
    fn y(&self) -> f64 {
        self.1.unwrap_or(f64::NAN)
    }
}

impl CoordXY for [f64; 2] {
    #[inline]
    fn x(&self) -> f64 {
        self[0]
    }
    #[inline]
    fn y(&self) -> f64 {
        self[1]
    }
}

#[cfg(feature = "num-complex")]
impl CoordXY for num_complex::Complex64 {
    #[inline]
    fn x(&self) -> f64 {
        self.re
    }
    #[inline]
    fn y(&self) -> f64 {
        self.im
    }
}

impl<'a, I> XYFrom<'a, I>
where
    I: IntoIterator,
    <I as IntoIterator>::Item: CoordXY,
{
    fn new(axes: &'a Axes, xy: I) -> Self {
        Self { axes, options: PlotOptions::new(), data: xy }
    }

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
        Python::attach(|py| {
            let x = x.to_pyvector(py);
            let y = y.to_pyvector(py);
            self.options.plot_xy(py, self.axes, &x, &y)
        })
    }
}

/// Options to plot functions (require the library [curve-sampling][]).
/// Created by [`Axes::fun`].
///
/// [curve-sampling]: https://crates.io/crates/curve-sampling
#[must_use]
pub struct Fun<'a, F, D> {
    axes: &'a Axes,
    options: PlotOptions<'a>,
    f: F,
    data: PhantomData<D>, // Data produced by `f`.
    a: f64,               // [a, b] is the interval on which we want to plot f.
    b: f64,
    n: usize,
}

#[cfg(feature = "curve-sampling")]
impl<'a, F, Y, D> Fun<'a, F, D>
where
    F: FnMut(f64) -> Y,
    Y: curve_sampling::Img<D>,
{
    fn new(axes: &'a Axes, f: F, a: f64, b: f64) -> Self {
        Self {
            axes, options: PlotOptions::new(), f, data: PhantomData,
            a, b, n: 100,
        }
    }

    set_plotoptions!();

    /// Plot the data with the options specified in [`XY`].
    pub fn plot(mut self) -> Line2D {
        let s = Sampling::fun(&mut self.f, self.a, self.b).n(self.n).build();
        // Ensure `x` and `y` live to the end of the call to "plot".
        let x = s.x();
        let y = s.y();
        Python::attach(|py| {
            let x = x.to_pyvector(py);
            let y = y.to_pyvector(py);
            self.options.plot_xy(py, self.axes, &x, &y)
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

/// Options for [`Axes::scatter`].
#[must_use]
pub struct Scatter<'a> {
    axes: &'a Axes,
    x: Py<PyArray1<f64>>,
    y: Py<PyArray1<f64>>,
    // Optional arguments are different from other plot types.
    s: Option<&'a [f64]>,
    c: Option<ScatterColorMat<'a>>, // Slice of RGBA
    marker: Option<&'a str>, // TODO: generalize
    cmap: Option<()>, // TODO
    norm: Option<()>, // TODO
    // FIXME: It is an error to use vmin/vmax when a norm instance is
    // given (but using a str norm name together with vmin/vmax is
    // acceptable).
    vmin: Option<f64>,
    vmax: Option<f64>,
    alpha: Option<f64>, // ∈ [0, 1]
    linewidths: Option<f64>, // TODO: support array-like
    // edgecolors
    // colorizer
    // plotnonfinite
}

impl<'a> Scatter<'a> {
    fn new(
        axes: &'a Axes,
        x: &'a (impl Vector<f64> + ?Sized),
        y: &'a (impl Vector<f64> + ?Sized),
    ) -> Self {
        Python::attach(|py| {
            let x = x.to_pyvector(py).unbind();
            let y = y.to_pyvector(py).unbind();
            Self {
                axes, x, y,
                s: None, c: None, marker: None, cmap: None, norm: None,
                vmin: None, vmax: None, alpha: None, linewidths: None,
            }
        })
    }

    /// The marker size in points² (typographic points are 1/72 in).
    ///
    /// Default is rcParams['lines.markersize'] ** 2.
    pub fn s(mut self, s: &'a [f64]) -> Self {
        self.s = Some(s);
        self
    }

    /// Specify the marker color(s).
    pub fn c<C>(mut self, colors: impl ScatterColors) -> Self
    where C: Color,
    {
        self.c = Some(ScatterColorMat::Colors(colors.as_mat()));
        self
    }

    /// Specify the marker colors as a sequence of `n` numbers to be
    /// mapped to colors using `cmap` and `norm` where `n` is the
    /// length of the data (see [`Axes::scatter`]).
    pub fn cm(mut self, colors: &'a [usize]) -> Self {
        self.c = Some(ScatterColorMat::Cmap(colors));
        self
    }

    /// Set the marker style.
    pub fn marker(mut self, m: &'a str) -> Self {
        self.marker = Some(m);
        self
    }

    /// When using scalar data and no explicit `norm`, `vmin` and
    /// [`vmax`][Scatter::vmax] define the data range that the
    /// colormap covers.
    pub fn vmin(mut self, v: f64) -> Self {
        self.vmin = Some(v);
        self
    }

    /// When using scalar data and no explicit `norm`,
    /// [`vmin`][Scatter::vmin] and `vmax` define the data range that
    /// the colormap covers.
    pub fn vmax(mut self, v: f64) -> Self {
        self.vmax = Some(v);
        self
    }

    /// Set the alpha blending value, between 0 (transparent) and 1
    /// (opaque).
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha.clamp(0., 1.));
        self
    }

    /// The linewidth of the marker edges.
    ///
    /// Note: The default `edgecolors` is "face".  You may want to
    /// change this as well.
    pub fn linewidths(mut self, lw: f64) -> Self {
        self.linewidths = Some(lw);
        self
    }

    pub fn plot(self) {
        // FIXME: Do we want to check that `x` and `y` have the same
        // dimension?  Better error message?
        Python::attach(|py| match self.c {
            Some(ScatterColorMat::Cmap(v)) => {
                self.plot_with_colors(py, v.to_pyarray(py));
            }
            Some(ScatterColorMat::Colors(ref m)) => {
                self.plot_with_colors(py, m.to_pyarray(py));
            }
            None => self.plot_with_colors(py, None::<&str>),
        })
    }

    fn plot_with_colors<'py>(
        &self,
        py: Python<'py>,
        c: impl IntoPyObject<'py>,
    ) {
        self.axes.ax.call_method1(py, intern!(py, "scatter"),
            (&self.x, &self.y, self.s, c, self.marker, self.cmap,
                 self.norm, self.vmin, self.vmax, self.alpha,
                 self.linewidths))
            .unwrap();
    }
}

enum ScatterColorMat<'a> {
    Cmap(&'a [usize]),
    Colors(ndarray::Array2<f64>),
}

/// Possible color specifications for [`Axes::scatter`] plots.
pub trait ScatterColors {
    #[doc(hidden)]
    fn as_mat(&self) -> ndarray::Array2<f64>;
}

impl<C> ScatterColors for &[C]
where
    C: Color,
{
    fn as_mat(&self) -> ndarray::Array2<f64> {
        let n = self.len();
        let mut c: Array2<f64> = ndarray::Array2::zeros((n, 4));
        for i in 0..n {
            let ci = self[i].rgba();
            for j in 0..4 {
                c[(i, j)] = ci[j];
            }
        }
        c
    }
}

impl<C: Color> ScatterColors for C {
    fn as_mat(&self) -> ndarray::Array2<f64> {
        // A single row array gives the same color for all markers.
        let mut c = ndarray::Array2::zeros((1, 4));
        let color = self.rgba();
        c[(0, 0)] = color[0];
        c[(0, 1)] = color[1];
        c[(0, 2)] = color[2];
        c[(0, 3)] = color[3];
        c
    }
}

/// Options for [`Axes::bar`].
pub struct Bar<'a> {
    axes: &'a Axes,
    x: Py<PyArray1<f64>>, // FIXME: categorical data ?
    height: Py<PyArray1<f64>>,
    width: f64, // FIXME: or array
    bottom: f64, // FIXME: or array
    align: BarAlign,
    // Options
    color: Option<[f64; 4]>, // FIXME: or array
    facecolor: Option<[f64; 4]>, // FIXME: or array
    edgecolor: Option<[f64; 4]>, // FIXME: or array
    linewidth: Option<f64>, // FIXME: or array
    tick_label: Option<&'a str>, // FIXME: or array
    label: Option<&'a str>, // FIXME: or array
    //xerr,  yerr, ecolor, capsize, error_kw, log
}

/// Alignment of [`Axes::bar`].  See [`Bar::align`].
#[derive(Debug, Clone, Copy)]
pub enum BarAlign {
    Center,
    Edge,
}

impl<'a> Bar<'a> {
    fn new(
        axes: &'a Axes,
        x: &'a (impl Vector<f64> + ?Sized),
        height: &'a (impl Vector<f64> + ?Sized),
    ) -> Self {
        Python::attach(|py| {
            let x = x.to_pyvector(py).unbind();
            let height = height.to_pyvector(py).unbind();
            Self {
                axes,
                x,
                height,
                width: 0.8,
                bottom: 0.,
                align: BarAlign::Center,
                color: None,  facecolor: None,  edgecolor: None,
                linewidth: None,  tick_label: None,  label: None,
            }
        })
    }

    pub fn width(mut self, w: f64) -> Self {
        self.width = w;
        self
    }

    pub fn bottom(mut self, b: f64) -> Self {
        self.bottom = b;
        self
    }

    pub fn align(mut self, a: BarAlign) -> Self {
        self.align = a;
        self
    }

    pub fn color(mut self, c: impl Color) -> Self {
        self.color = Some(c.rgba());
        self
    }

    pub fn facecolor(mut self, c: impl Color) -> Self {
        self.facecolor = Some(c.rgba());
        self
    }

    pub fn edgecolor(mut self, c: impl Color) -> Self {
        self.edgecolor = Some(c.rgba());
        self
    }

    pub fn linewidth(mut self, w: f64) -> Self {
        self.linewidth = Some(w);
        self
    }

    pub fn tick_label(mut self, l: &'a str) -> Self {
        self.tick_label = Some(l);
        self
    }

    pub fn label(mut self, l: &'a str) -> Self {
        self.label = Some(l);
        self
    }

    pub fn plot(self) {
        Python::attach(|py| {
            let align = match self.align {
                BarAlign::Center => "center",
                BarAlign::Edge => "edge",
            };
            let kwargs = PyDict::new(py);
            kwargs.set_item("align", align).unwrap();
            if let Some(color) = self.color {
                kwargs.set_item("color", color).unwrap();
            }
            if let Some(facecolor) = self.facecolor {
                kwargs.set_item("facecolor", facecolor).unwrap();
            }
            if let Some(edgecolor) = self.edgecolor {
                kwargs.set_item("edgecolor", edgecolor).unwrap();
            }
            if let Some(lw) = self.linewidth {
                kwargs.set_item("linewidth", lw).unwrap();
            }
            if let Some(tick_label) = self.tick_label {
                kwargs.set_item("tick_label", tick_label).unwrap();
            }
            if let Some(label) = self.label {
                kwargs.set_item("label", label).unwrap();
            }
            // TODO: options
            self.axes.ax.bind(py)
                .call_method(intern!(py, "bar"),
                    (self.x, self.height, self.width, self.bottom),
                    Some(&kwargs))
                .unwrap();
        })
    }
}

/// Options for [`Axes::stem`].
pub struct Stem<'a> {
    axes: &'a Axes,
    x: Py<PyArray1<f64>>,
    y: Py<PyArray1<f64>>,
    linefmt: &'a str, // FIXME: enum
    markerfmt: &'a str,
    basefmt: &'a str,
}

impl<'a> Stem<'a> {
    fn new(
        axes: &'a Axes,
        x: &'a (impl Vector<f64> + ?Sized),
        y: &'a (impl Vector<f64> + ?Sized),
    ) -> Self {
        Python::attach(|py| {
            let x = x.to_pyvector(py).unbind();
            let y = y.to_pyvector(py).unbind();
            Self {
                axes, x, y,
                linefmt: "C0-",
                markerfmt: "o",
                basefmt: "C3-",
            }
        })
    }

    pub fn linefmt(mut self, fmt: &'a str) -> Self {
        self.linefmt = fmt;
        self
    }

    pub fn markerfmt(mut self, fmt: &'a str) -> Self {
        self.markerfmt = fmt;
        self
    }

    pub fn basefmt(mut self, fmt: &'a str) -> Self {
        self.basefmt = fmt;
        self
    }

    pub fn plot(self) {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("linefmt", self.linefmt).unwrap();
            kwargs.set_item("markerfmt", self.markerfmt).unwrap();
            kwargs.set_item("basefmt", self.basefmt).unwrap();
            self.axes.ax.bind(py)
                .call_method(intern!(py, "stem"),
                    (self.x, self.y),
                    Some(&kwargs))
                .unwrap();
        })
    }
}

/// Options for [`Axes::fill_between`].
pub struct FillBetween<'a> {
    axes: &'a Axes,
    x: Py<PyArray1<f64>>,
    y1: Py<PyArray1<f64>>,
    y2: Py<PyArray1<f64>>,
    where_: Option<&'a [bool]>,
    interpolate: bool,
    step: Option<Step>,
    // Options.  TODO: may other — and share with FillBetweenPolyCollection
    alpha: Option<f64>, // or array
    linewidth: Option<f64>,
}

/// Possible values for [`FillBetween::step`].
#[derive(Debug, Clone, Copy)]
pub enum Step {
    Pre,
    Post,
    Mid,
}

impl Step {
    fn as_str(self) -> &'static str {
        match self {
            Step::Pre => "pre",
            Step::Post => "post",
            Step::Mid => "mid",
        }
    }
}

impl<'a> FillBetween<'a> {
    fn new(
        axes: &'a Axes,
        x: &'a (impl Vector<f64> + ?Sized),
        y1: &'a (impl Vector<f64> + ?Sized),
        y2: &'a (impl Vector<f64> + ?Sized),
    ) -> Self {
        Python::attach(|py| {
            let x = x.to_pyvector(py).unbind();
            let y1 = y1.to_pyvector(py).unbind();
            let y2 = y2.to_pyvector(py).unbind();
            Self {
                axes,
                x,
                y1, // or f64
                y2, // or f64
                where_: None,
                interpolate: false,
                step: None,
                alpha: None,
                linewidth: None,
            }
        })
    }

    pub fn step(mut self, s: Step) -> Self {
        self.step = Some(s);
        self
    }

    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = Some(a);
        self
    }

    pub fn linewidth(mut self, lw: f64) -> Self {
        self.linewidth = Some(lw);
        self
    }

    pub fn plot(self) {
        Python::attach(|py| {
            let step = self.step.map(Step::as_str);
            let kwargs = PyDict::new(py);
            if let Some(alpha) = self.alpha {
                kwargs.set_item("alpha", alpha).unwrap();
            }
            if let Some(lw) = self.linewidth {
                kwargs.set_item("linewidth", lw).unwrap();
            }
            self.axes.ax.bind(py)
                .call_method(intern!(py, "fill_between"),
                    (self.x, self.y1, self.y2,
                     self.where_, self.interpolate, step),
                    Some(&kwargs))
                .unwrap();
        })
    }
}

/// Options for [`Axes::stack`].
pub struct Stack<'a> {
    axes: &'a Axes,
    x: Py<PyArray1<f64>>,
    y: &'a Array2<f64>,
    // FIXME: there are optional arguments.
}

impl<'a> Stack<'a> {
    fn new(
        axes: &'a Axes,
        x: &'a (impl Vector<f64> + ?Sized),
        y: &'a Array2<f64>,
    ) -> Self {
        Python::attach(|py| {
            let x = x.to_pyvector(py).unbind();
            Self { axes, x, y }
        })
    }

    pub fn plot(self) {
        Python::attach(|py| {
            let y = self.y.to_pyarray(py);
            self.axes.ax.bind(py)
                .call_method(intern!(py, "stackplot"),
                    (self.x, y),
                    None)
                .unwrap();
        })
    }
}

/// Options for [`Axes::stairs`].
pub struct Stairs<'a> {
    axes: &'a Axes,
    y: Py<PyArray1<f64>>,
    edges: Option<Py<PyArray1<f64>>>,
    orientation: Orientation,
    fill: bool,
    // FIXME: there are more optional arguments.
    linewidth: Option<f64>,
}

/// Orientation for [`Stairs::orientation`].
#[derive(Debug, Clone, Copy)]
pub enum Orientation {
    Horizontal,
    Vertical,
}

impl Orientation {
    fn as_str(self) -> &'static str {
        match self {
            Orientation::Horizontal => "horizontal",
            Orientation::Vertical => "vertical",
        }
    }
}

impl<'a> Stairs<'a> {
    fn new(axes: &'a Axes, y: &'a (impl Vector<f64> + ?Sized)) -> Self {
        Python::attach(|py| {
            let y = y.to_pyvector(py).unbind();
            Self {
                axes, y,
                edges: None,
                orientation: Orientation::Vertical,
                fill: false,
                linewidth: None,
            }
        })
    }

    /// Define the x-axis positions of the steps.
    pub fn edges(mut self, x: &'a (impl Vector<f64> + ?Sized)) -> Self {
        Python::attach(|py| {
            let x = x.to_pyvector(py).unbind();
            self.edges = Some(x);
        });
        self
    }

    pub fn orientation(mut self, o: Orientation) -> Self {
        self.orientation = o;
        self
    }

    pub fn fill(mut self) -> Self {
        self.fill = true;
        self
    }

    pub fn linewidth(mut self, lw: f64) -> Self {
        self.linewidth = Some(lw);
        self
    }

    pub fn plot(self) {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("orientation", self.orientation.as_str()).unwrap();
            kwargs.set_item("fill", self.fill).unwrap();
            if let Some(lw) = self.linewidth {
                kwargs.set_item("linewidth", lw).unwrap();
            }
            self.axes.ax.bind(py)
                .call_method(intern!(py, "stairs"),
                    (self.y, self.edges),
                    Some(&kwargs))
                .unwrap();
        })
    }
}

/// Set of contour lines or filled regions.
///
/// Returned by [`Axes::contour`] and [`Axes::contour_fun`].
pub struct QuadContourSet {
    contours: Py<PyAny>,
}

impl QuadContourSet {
    pub fn set_color(&mut self, c: impl Color) -> &mut Self {
        Python::attach(|py| {
            meth!(self.contours, set_color, (colors::py(py, c),)).unwrap()
        });
        self
    }
}

macro_rules! set_contour_options {
    () => {
        pub fn levels(mut self, levels: &'a [f64]) -> Self {
            self.levels = Some(levels);
            self
        }

        pub fn colors<C: Color>(mut self, colors: impl AsRef<[C]>) -> Self {
            let colors = colors.as_ref();
            let mut rgbas = Vec::with_capacity(colors.len());
            for c in colors {
                rgbas.push(c.rgba());
            }
            self.colors = Some(rgbas);
            self
        }

        fn update_dict(&self, d: &mut Bound<PyDict>) {
            let py = d.py();
            if let Some(levels) = self.levels {
                let n = levels.len();
                let levels = levels.to_pyarray(py);
                d.set_item("levels", levels).unwrap();

                if let Some(colors) = &self.colors {
                    if colors.len() >= n {
                        let colors = PyList::new(py, colors).unwrap();
                        d.set_item("colors", colors).unwrap();
                    } else {
                        let default = self.options.color.unwrap_or([0., 0., 0., 1.]);
                        let mut colors = colors.clone();
                        for _ in 0..n - colors.len() {
                            colors.push(default);
                        }
                        let colors = PyList::new(py, colors).unwrap();
                        d.set_item("colors", colors).unwrap();
                    }
                } else if let Some(color) = self.options.color {
                    // let colors = std::iter::repeat_n(color, n);
                    let colors = vec![color; n];
                    let colors = PyList::new(py, colors).unwrap();
                    d.set_item("colors", colors).unwrap();
                }
            }
        }
    };
}

/// Options for [`Axes::contour`].
#[must_use]
pub struct Contour<'a> {
    axes: &'a Axes,
    options: PlotOptions<'a>,
    x: Py<PyArray1<f64>>,
    y: Py<PyArray1<f64>>,
    z: &'a ndarray::Array2<f64>,
    levels: Option<&'a [f64]>,
    colors: Option<Vec<[f64; 4]>>,
}

impl<'a> Contour<'a> {
    fn new(
        axes: &'a Axes,
        x: &'a (impl Vector<f64> + ?Sized),
        y: &'a (impl Vector<f64> + ?Sized),
        z: &'a ndarray::Array2<f64>,
    ) -> Self {
        Python::attach(|py| {
            let x = x.to_pyvector(py).unbind();
            let y = y.to_pyvector(py).unbind();
            Self {
                axes, options: PlotOptions::new(),
                x, y, z, levels: None, colors: None,
            }
        })
    }

    set_plotoptions!();
    set_contour_options!();

    pub fn plot(&self) -> QuadContourSet {
        Python::attach(|py| {
            let z = self.z.to_pyarray(py);
            let mut opt = self.options.kwargs(py);
            self.update_dict(&mut opt);
            let contours = self.axes.ax
                .call_method(py, intern!(py, "contour"),
                    (&self.x, &self.y, z),
                    Some(&opt))
                .unwrap();
            QuadContourSet { contours }
        })
    }
}

/// Options for [`Axes::contour_fun`].
#[must_use]
pub struct ContourFun<'a, F> {
    axes: &'a Axes,
    options: PlotOptions<'a>,
    f: F,
    ab: [f64; 2],
    cd: [f64; 2],
    n1: usize, // FIXME: want to be more versatile than an equispaced grid?
    n2: usize,
    levels: Option<&'a [f64]>,
    colors: Option<Vec<[f64; 4]>>,
}

impl<'a, F> ContourFun<'a, F>
where
    F: FnMut(f64, f64) -> f64,
{
    fn new(axes: &'a Axes, ab: [f64; 2], cd: [f64; 2], f: F) -> Self {
        Self {
            axes, options: PlotOptions::new(),
            f, ab, cd, n1: 100, n2: 100,
            levels: None, colors: None,
        }
    }

    set_plotoptions!();
    set_contour_options!();

    pub fn plot(&mut self) -> QuadContourSet {
        let mut x = Vec::with_capacity(self.n1);
        let mut y = Vec::with_capacity(self.n2);
        let mut z = ndarray::Array2::zeros((self.n2, self.n1));
        let a = self.ab[0];
        let dx = (self.ab[1] - a) / (self.n1 - 1) as f64;
        for i in 0..self.n1 {
            x.push(a + dx * i as f64);
        }
        let c = self.cd[0];
        let dy = (self.cd[1] - c) / (self.n2 - 1) as f64;
        for j in 0..self.n2 {
            y.push(c + dy * j as f64);
        }
        for (j, &y) in y.iter().enumerate() {
            for (i, &x) in x.iter().enumerate() {
                z[(j, i)] = (self.f)(x, y);
            }
        }
        Python::attach(|py| {
            let x = x.to_pyarray(py);
            let y = y.to_pyarray(py);
            let z = z.to_pyarray(py);
            let mut opt = self.options.kwargs(py);
            self.update_dict(&mut opt);
            let contours = self
                .axes
                .ax
                .call_method(py, intern!(py, "contour"), (x, y, z), Some(&opt))
                .unwrap();
            QuadContourSet { contours }
        })
    }
}


#[cfg(test)]
mod test {
    use crate::figure::Figure;

    #[test]
    fn test_unsized() -> Result<(), crate::Error> {
        let fig = Figure::new()?;
        let [[mut ax]] = fig.subplots()?;
        let x = [0., 1., 2., 3.];
        // The slice `[f64]` is unsized.
        ax.xy(&x[..], &x[..]).plot();
        ax.y(&x[..]).plot();
        fig.save().to_file("target/test_axes_unsized.pdf")?;
        Ok(())
    }

    #[test]
    fn test_get_xticklabels() -> Result<(), crate::Error> {
        let fig = Figure::new()?;
        let [[mut ax]] = fig.subplots()?;
        ax.xy(&[0., 1.], &[0., 1.]).plot();
        for l in ax.get_xticklabels() {
            l.set_rotation(45.);
        }
        fig.save().to_file("target/test_get_xticklabels.pdf")?;
        Ok(())
    }

    #[test]
    fn test_minorticks_on() -> Result<(), crate::Error> {
        let fig = Figure::new()?;
        let [[mut ax]] = fig.subplots()?;
        ax.minorticks_on().grid();
        ax.xy(&[0., 1.], &[0., 1.]).plot();
        fig.save().to_file("target/test_minorticks_on.pdf")?;
        Ok(())
    }
}
