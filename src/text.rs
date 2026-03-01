//! [Text](https://matplotlib.org/stable/api/text_api.html) included
//! in a figure.

use std::convert::Infallible;

use pyo3::{intern, prelude::*, sync::PyOnceLock};

pub struct Text {
    text: Py<PyAny>,
}

impl FromPyObject<'_, '_> for Text {
    type Error = Infallible;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        Ok(Self {
            text: obj.to_owned().unbind(),
        })
    }
}

/// Text alignment specification.
#[derive(Clone, Copy, Debug)]
pub enum Align {
    Left,
    Center,
    Right,
}

impl Align {
    #[inline]
    fn as_str(&self) -> &str {
        match self {
            Align::Left => "left",
            Align::Center => "center",
            Align::Right => "right",
        }
    }
}

macro_rules! set {
    ($(#[$doc: meta])* $name: ident $(, $v: ident, $t: ty, $conv: expr)*) => {
        $(#[$doc])*
        pub fn $name(&self, $($v: $t)*) -> &Self {
            Python::attach(|py| {
                self.text.bind(py)
                .call_method1(stringify!($name), ($($conv,)*))
                .unwrap();
            });
            self
    }}}

impl Text {
    fn cls(py: Python<'_>) -> &Bound<'_, PyAny> {
        static TEXT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
        TEXT.import(py, "matplotlib.text", "Text")
            .expect("Cannot find matplotlib.text.Text")
    }

    // FIXME: x: impl Coord
    pub fn new(x: f64, y: f64, text: &str) -> Self {
        Python::attach(|py| {
            Self::cls(py)
                .call1((x, y, text))
                .expect("New Text")
                .unbind()
                .extract(py)
                .unwrap()
        })
    }

    /// Return whether antialiased rendering is used.
    pub fn get_antialiased(&self) -> bool {
        Python::attach(|py| {
            self.text
                .call_method1(py, intern!(py, "get_antialiased"), ())
                .unwrap()
                .extract(py)
                .unwrap()
        })
    }

    set!(
        /// Set the horizontal alignment relative to the anchor point.
        set_horizontalalignment, align, Align, align.as_str());

    set!(
        /// Set the rotation of the text.
        set_rotation, s, f64, s);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_antialiased() {
        let _ = Text::new(0., 0., "Hello").get_antialiased();
    }
}
