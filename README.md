Matplotlib.rs
=============

This is a [Matplotlib][] binding for Rust.  Its interface tries to be
similar to the Matplotlib one (so users knowing the Python version can
easily search for the corresponding functions) but deviating whenever
it makes sense (for example, to specify optional arguments) to have a
nice Rust interface.  Data is _shared_ between Rust and Python (no
copying, no temporary files).

This library is _work in progress_.


Non Rust dependencies
---------------------

The binding is made using [PyO3][], thus you “[need to ensure that
your Python installation contains a shared library][shared-lib]”.  Of
course you also need [Matplotlib][] to be installed.


Example
-------

```rust
let (fig, [[mut ax]]) = Plot::sub()?;
ax.plot(&[1., 2., 3., 4.], &[1., 4., 2., 3.], "");
fig.savefig("plot.pdf")?;
```


![Basic Example](https://matplotlib.org/stable/_images/sphx_glr_quick_start_001_2_0x.png)


[Matplotlib]: https://matplotlib.org/
[IntoIterator]: https://doc.rust-lang.org/std/iter/trait.IntoIterator.html
[PyO3]: https://crates.io/crates/pyo3
[shared-lib]: https://crates.io/crates/pyo3#user-content-using-python-from-rust
