#[macro_export]
macro_rules! scan_helper {
    ($bar:ident,) => {$bar.inc(1);};
    ($bar:ident, [$name:ident = $iter:expr] $($rest:tt)*) => {
        rayon::iter::IntoParallelIterator::into_par_iter($iter).for_each(|$name| {
        pon::scan_helper! { $bar, $($rest)* }
        })
    };
    ($bar:ident, $s:expr; $($rest:tt)*) => {
        $s;
        pon::scan_helper! { $bar, $($rest)* }
    };
    ($bar:ident, $s:stmt; $($rest:tt)*) => {
        $s;
        pon::scan_helper! { $bar, $($rest)* }
    };
}

#[macro_export]
macro_rules! size_helper {
    () => { 1 };
    ([$name:ident = $iter:expr] $($rest:tt)*) => {
        //pon::size_helper! { $($rest)* }
        rayon::iter::IntoParallelIterator::into_par_iter($iter).opt_len().unwrap() * (pon::size_helper! { $($rest)* })
    };
    ($s:expr; $($rest:tt)*) => {
        pon::size_helper! { $($rest)* }
    };
    ($s:stmt; $($rest:tt)*) => {
        pon::size_helper! { $($rest)* }
    };
}

#[macro_export]
macro_rules! scan {
    ($($rest:tt)*) => {
        use rayon::iter::ParallelIterator;
        let size = pon::size_helper!($($rest)*);
        let mut progress_bar = indicatif::ProgressBar::new(size as u64);
        progress_bar.set_style(indicatif::ProgressStyle::default_bar().template("[{elapsed_precise}] {wide_bar} {pos}/{len} {eta}").unwrap());
        progress_bar.tick();
        pon::scan_helper! { progress_bar, $($rest)* }
    };
}

#[cfg(test)]
mod tests {
    /*#[test]
    fn test_scan() {
        let streamer = Streamer::new(Path::new("/tmp/x")).unwrap();
        crate::scan! {
            let x = 10;
            [n = 0..10]
            [m = 0..n]
            streamer.send(&format!("a {} {}", x, m));
        }
        streamer.join();
    }*/
}
