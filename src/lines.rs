//!

use crate::{
    colors::{self, Color},
};
use pyo3::{intern, prelude::*, types::PyDict};

include!("macros.rs");

/// A line — the line can have both a solid linestyle connecting all
/// the vertices, and a marker at each vertex. Additionally, the
/// drawing of the solid line is influenced by the drawstyle, e.g.,
/// one can create "stepped" lines in various styles.
pub struct Line2D {
    pub(crate) line2d: Py<PyAny>,
}

impl Line2D {
    fn set_kw<'a>(&self, py: Python<'a>, prop: &str, v: impl IntoPyObject<'a>) {
        let kwargs = PyDict::new(py);
        kwargs.set_item(prop, v).unwrap();
        self.line2d
            .call_method(py, "set", (), Some(&kwargs))
            .unwrap();
    }

    pub fn set_label(&mut self, label: impl AsRef<str>) -> &mut Self {
        Python::attach(|py| {
            self.set_kw(py, "label", label.as_ref());
            self
        })
    }

    /// Set the color of the line to `c`.
    pub fn set_color(&mut self, c: impl Color) -> &mut Self {
        Python::attach(|py| {
            meth!(self.line2d, set_color, (colors::py(py, c),)).unwrap();
            self
        })
    }

    pub fn set_linewidth(&mut self, w: f64) -> &mut Self {
        Python::attach(|py| {
            self.set_kw(py, "linewidth", w);
            self
        })
    }

    pub fn linewidth(self, w: f64) -> Self {
        Python::attach(|py| {
            self.set_kw(py, "linewidth", w);
            self
        })
    }
}
