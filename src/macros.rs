#![macro_use]

#[macro_export]
macro_rules! getattr {
    ($py: ident, $lib: expr, $f: literal) => {
        $lib.getattr($py, intern!($py, $f)).unwrap()
    };
}

#[macro_export]
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

/// Import and return a handle to the module `$m`.
#[macro_export]
macro_rules! pyimport { ($name: path, $m: literal) => {
    Python::attach(|py|
        match PyModule::import(py, intern!(py, $m)) {
            Ok(m) => Ok(m.into()),
            Err(e) => {
                let mut msg = stringify!($name).to_string();
                msg.push_str(": ");
                if let Ok(s) = e.value(py).str() {
                    let s = s.to_str().unwrap_or("Import error");
                    msg.push_str(s)
                }
                Err(ImportError(msg))
            }
        })
}}
