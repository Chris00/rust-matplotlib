
#[allow(unused_macros)]
macro_rules! getattr {
    ($py: ident, $lib: expr, $f: literal) => {
        $lib.getattr($py, intern!($py, $f)).unwrap()
    };
}

#[allow(unused_macros)]
macro_rules! meth {
    ($obj: expr, $m: ident, $py: ident -> $args: expr) => {
        Python::attach(|py| {
            let $py = py;
            $obj.call_method1(py, intern!(py, stringify!($m)), $args)
        })
    };
    ($obj: expr, $m: ident, $args: expr) => {
        Python::attach(|py| {
            $obj.call_method1(py, intern!(py, stringify!($m)), $args)
        })
    };
}
