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
use numpy::{Ix1, convert::ToPyArray};
use pyo3::{
    intern,
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use ndarray::Array2;
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

/// Alias for types convertible to one-dimensional Python arrays.
pub trait Vector<T>: ToPyArray<Item=T, Dim=Ix1> {}

impl<T, X> Vector<T> for X where X: ToPyArray<Item=T, Dim=Ix1> {}

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
    pub fn xy<'a, D>(&'a mut self, x: D, y: D) -> XY<'a, D>
    where
        D: AsRef<[f64]>,
    {
        // The chain leading to plot starts with the data (using this
        // function) so that additional data may be added, sharing
        // common options.  We also mutably borrow `self` to reflect that
        // the final `.plot()` will mutate the underlying Python object.
        XY {
            axes: self,
            options: PlotOptions::new(),
            data: PlotData::XY(x, y),
        }
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
    pub fn y<'a, D>(&'a mut self, y: D) -> XY<'a, D>
    where
        D: AsRef<[f64]>,
    {
        XY {
            axes: self,
            options: PlotOptions::new(),
            data: PlotData::Y(y),
        }
    }

    /// Convenience function to plot X-Y coordinates coming from `xy`.
    ///
    /// # Example
    ///
    /// ```
    /// use matplotlib::pyplot as plt;
    /// let (fig, [[mut ax]]) = plt::subplots()?;
    /// ax.xy_from(&[(1., 2.), (4., 2.), (2., 3.), (3., 4.)]).plot();
    /// ax.xy_from([(1., 0.), (2., 3.), (3., 1.), (4., 3.)]).plot();
    /// fig.save().to_file("target/XY_from_plot.pdf")?;
    /// # Ok::<(), matplotlib::Error>(())
    /// ```
    // FIXME: show an example combining iterators using `zip`.  The
    // `covid` research project may serve as a source of inspiration.
    pub fn xy_from<'a, I>(&'a mut self, xy: I) -> XYFrom<'a, I>
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: CoordXY,
    {
        XYFrom {
            axes: self,
            options: PlotOptions::new(),
            data: xy,
        }
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
        Fun {
            axes: self,
            options: PlotOptions::new(),
            f,
            data: PhantomData,
            a,
            b,
            n: 100,
        }
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
    /// ax.contour(x.as_slice().unwrap(), y.as_slice().unwrap(), &z)
    ///     .levels(&[0.2, 0.5, 0.8])
    ///     .colors(&[Tab::Red, Tab::Blue, Tab::Olive])
    ///     .plot();
    /// fig.save().to_file("target/contour.pdf")?;
    /// # Ok::<(), matplotlib::Error>(())
    /// ```
    pub fn contour<'a, D>(&'a mut self, x: D, y: D, z: &'a ndarray::Array2<f64>) -> Contour<'a, D>
    where
        D: AsRef<[f64]>,
    {
        Contour {
            axes: self,
            options: PlotOptions::new(),
            x,
            y,
            z,
            levels: None,
            colors: None,
        }
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
    pub fn contour_fun<'a, F>(&'a mut self, ab: [f64; 2], cd: [f64; 2], f: F) -> ContourFun<'a, F>
    where
        F: FnMut(f64, f64) -> f64,
    {
        ContourFun {
            axes: self,
            options: PlotOptions::new(),
            f,
            ab,
            cd,
            n1: 100,
            n2: 100,
            levels: None,
            colors: None,
        }
    }

    /// Scatter plot of `y` vs. `x` with optional varying marker size
    /// and/or color.
    pub fn scatter<'a, D>(&'a mut self, x: D, height: D) -> Scatter<'a, D>
    where
        D: AsRef<[f64]>,
    {
        Scatter::new(self, x, height)
    }

    /// Make a bar plot.
    ///
    /// The bars are positioned at `x` with the given
    /// [alignment][`Bar::align`].  Their dimensions are given by
    /// height and width. The vertical baseline is bottom (default 0).
    pub fn bar<'a, D1, D2>(&'a mut self, x: D1, height: D2) -> Bar<'a>
    where
        D1: AsRef<[f64]> + 'a,
        D2: AsRef<[f64]> + 'a,
    {
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
    pub fn stem<'a, D1, D2>(&'a mut self, x: D1, y: D2) -> Stem<'a>
    where
        D1: AsRef<[f64]> + 'a,
        D2: AsRef<[f64]> + 'a,
    {
        Stem::new(self, x, y)
    }

    pub fn fill_between<'a>(
        &'a mut self,
        x: &'a impl Vector<f64>,
        y1: &'a impl Vector<f64>,
        y2: &'a impl Vector<f64>,
    ) -> FillBetween<'a> {
        FillBetween::new(self, x, y1, y2)
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
            let labels: Vec<_> = self.ax.bind(py)
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

enum PlotData<D> {
    XY(D, D),
    Y(D),
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
            label: &"",
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
    fn plot_xy(&self, py: Python<'_>, axes: &Axes, x: &[f64], y: &[f64]) -> Line2D {
        let x = x.to_pyarray(py);
        let y = y.to_pyarray(py);
        let lines = axes
            .ax
            .call_method(py, "plot", (x, y, self.fmt), Some(&self.kwargs(py)))
            .unwrap();
        let lines: &Bound<PyList> = lines.cast_bound(py).unwrap();
        // Extract the element from the list of length 1 (1 data plotted)
        let line2d = lines.get_item(0).unwrap().into();
        Line2D { line2d }
    }

    fn plot_y(&self, py: Python<'_>, axes: &Axes, y: &[f64]) -> Line2D {
        let y = y.to_pyarray(py);
        let lines = axes
            .ax
            .call_method(py, "plot", (y, self.fmt), Some(&self.kwargs(py)))
            .unwrap();
        let lines: &Bound<PyList> = lines.cast_bound(py).unwrap();
        let line2d = lines.get_item(0).unwrap().into();
        Line2D { line2d }
    }

    fn plot_data<D: AsRef<[f64]>>(
        &self,
        py: Python<'_>,
        axes: &Axes,
        data: &PlotData<D>,
    ) -> Line2D {
        match data {
            PlotData::XY(x, y) => self.plot_xy(py, axes, x.as_ref(), y.as_ref()),
            PlotData::Y(y) => self.plot_y(py, axes, y.as_ref()),
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
pub struct XY<'a, D> {
    axes: &'a Axes,
    options: PlotOptions<'a>,
    data: PlotData<D>,
}

impl<'a, D> XY<'a, D>
where
    D: AsRef<[f64]>,
{
    set_plotoptions!();

    /// Plot the data with the options specified in [`XY`].
    pub fn plot(self) -> Line2D {
        Python::attach(|py| self.options.plot_data(py, self.axes, &self.data))
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
        Python::attach(|py| self.options.plot_xy(py, self.axes, &x, &y))
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
    set_plotoptions!();

    /// Plot the data with the options specified in [`XY`].
    pub fn plot(mut self) -> Line2D {
        let s = Sampling::fun(&mut self.f, self.a, self.b).n(self.n).build();
        // Ensure `x` and `y` live to the end of the call to "plot".
        let x = s.x();
        let y = s.y();
        Python::attach(|py| self.options.plot_xy(py, self.axes, &x, &y))
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
pub struct Scatter<'a, D> {
    axes: &'a Axes,
    x: D,
    y: D,
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

impl<'a, D> Scatter<'a, D>
where D: AsRef<[f64]> {
    fn new(axes: &'a Axes, x: D, y: D) -> Self {
        Self {
            axes, x, y,
            s: None, c: None, marker: None, cmap: None, norm: None,
            vmin: None, vmax: None, alpha: None, linewidths: None,
        }
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
    /// [`vmax`] define the data range that the colormap covers.
    pub fn vmin(mut self, v: f64) -> Self {
        self.vmin = Some(v);
        self
    }

    /// When using scalar data and no explicit `norm`, [`vmin`] and
    /// `vmax` define the data range that the colormap covers.
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
        Python::attach(|py| {
            match self.c {
                Some(ScatterColorMat::Cmap(v)) => {
                    self.plot_with_colors(py, v.to_pyarray(py));
                }
                Some(ScatterColorMat::Colors(ref m)) => {
                    self.plot_with_colors(py, m.to_pyarray(py));
                }
                None => self.plot_with_colors(py, None::<&str>),
            }
        })
    }

    fn plot_with_colors<'py>(
        &self,
        py: Python<'py>,
        c: impl IntoPyObject<'py>,
    ) {
        let xn = self.x.as_ref().to_pyarray(py);
        let yn = self.y.as_ref().to_pyarray(py);
        self.axes.ax.call_method1(py, intern!(py, "scatter"),
            (xn, yn, self.s, c, self.marker, self.cmap,
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
        let colors = self.as_ref();
        let n = colors.len();
        let mut c: Array2<f64> = ndarray::Array2::zeros((n, 4));
        for i in 0 .. n {
            let ci = colors[i].rgba();
            for j in 0 .. 4 {
                c[(i,j)] = ci[j];
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
        c[(0,0)] = color[0];
        c[(0,1)] = color[1];
        c[(0,2)] = color[2];
        c[(0,3)] = color[3];
        c
    }
}

pub struct Bar<'a> {
    axes: &'a Axes,
    x: Box<dyn AsRef<[f64]> + 'a>, // FIXME: categorical data ?
    height: Box<dyn AsRef<[f64]> + 'a>,
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
    fn new<D1, D2>(axes: &'a Axes, x: D1, height: D2) -> Self
    where
        D1: AsRef<[f64]> + 'a,
        D2: AsRef<[f64]> + 'a,
    {
        Self {
            axes,
            x: Box::new(x),
            height: Box::new(height),
            width: 0.8,
            bottom: 0.,
            align: BarAlign::Center,
            color: None,  facecolor: None,  edgecolor: None,
            linewidth: None,  tick_label: None,  label: None,
        }
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
            let x = self.x.as_ref().as_ref().to_pyarray(py);
            let height = self.height.as_ref().as_ref().to_pyarray(py);
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
                    (x, height, self.width, self.bottom),
                    Some(&kwargs))
                .unwrap();
        })
    }
}

pub struct Stem<'a> {
    axes: &'a Axes,
    x: Box<dyn AsRef<[f64]> + 'a>,
    y: Box<dyn AsRef<[f64]> + 'a>,
    linefmt: &'a str, // FIXME: enum
    markerfmt: &'a str,
    basefmt: &'a str,
}

impl<'a> Stem<'a> {
    fn new<D1, D2>(axes: &'a Axes, x: D1, y: D2) -> Self
    where
        D1: AsRef<[f64]> + 'a,
        D2: AsRef<[f64]> + 'a,
    {
        Self {
            axes,
            x: Box::new(x),
            y: Box::new(y),
            linefmt: "C0-",
            markerfmt: "o",
            basefmt: "C3-",
        }
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
            let x = self.x.as_ref().as_ref().to_pyarray(py);
            let y = self.y.as_ref().as_ref().to_pyarray(py);
            let kwargs = PyDict::new(py);
            kwargs.set_item("linefmt", self.linefmt).unwrap();
            kwargs.set_item("markerfmt", self.markerfmt).unwrap();
            kwargs.set_item("basefmt", self.basefmt).unwrap();
            self.axes.ax.bind(py)
                .call_method(intern!(py, "stem"),
                    (x, y),
                    Some(&kwargs))
                .unwrap();
        })
    }
}

pub struct FillBetween<'a> {
    axes: &'a Axes,
    x: &'a dyn Vector<f64>,
    y1: &'a dyn Vector<f64>,
    y2: &'a dyn Vector<f64>,
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
    fn new<D0, D1, D2>(axes: &'a Axes, x: &'a D0, y1: &'a D1, y2: &'a D2) -> Self
    where
        D0: Vector<f64> + 'a,
        D1: Vector<f64> + 'a,
        D2: Vector<f64> + 'a,
    {
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
            let x = self.x.to_pyarray(py);
            let y1 = self.y1.to_pyarray(py);
            let y2 = self.y2.to_pyarray(py);
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
                    (x, y1, y2, self.where_, self.interpolate, step),
                    Some(&kwargs))
                .unwrap();
        })
    }
}

pub struct QuadContourSet {
    contours: Py<PyAny>,
}

impl QuadContourSet {
    pub fn set_color(&mut self, c: impl Color) -> &mut Self {
        Python::attach(|py| meth!(self.contours, set_color, (colors::py(py, c),)).unwrap());
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

#[must_use]
pub struct Contour<'a, D> {
    axes: &'a Axes,
    options: PlotOptions<'a>,
    x: D,
    y: D,
    z: &'a ndarray::Array2<f64>,
    levels: Option<&'a [f64]>,
    colors: Option<Vec<[f64; 4]>>,
}

impl<'a, D> Contour<'a, D>
where
    D: AsRef<[f64]>,
{
    set_plotoptions!();
    set_contour_options!();

    pub fn plot(&self) -> QuadContourSet {
        Python::attach(|py| {
            let x = self.x.as_ref().to_pyarray(py);
            let y = self.y.as_ref().to_pyarray(py);
            let z = self.z.to_pyarray(py);
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
        ax.minorticks_on()
            .grid();
        ax.xy(&[0., 1.], &[0., 1.]).plot();
        fig.save().to_file("target/test_minorticks_on.pdf")?;
        Ok(())
    }
}
