use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

struct Inner {
    writer: BufWriter<File>,
}

pub struct Streamer {
    inner: Mutex<Inner>,
}

impl Streamer {
    pub fn new(path: &Path) -> std::io::Result<Self> {
        let writer = BufWriter::new(File::create(path)?);
        Ok(Streamer {
            inner: Mutex::new(Inner { writer }),
        })
    }

    pub fn write_str(&self, value: &str) {
        let mut inner = self.inner.lock().unwrap();
        writeln!(inner.writer, "{}", value).unwrap();
    }

    pub fn send<T: Serialize>(&self, value: &T) {
        let serialized_value: String = serde_json::to_string(value).unwrap();
        self.write_str(&serialized_value);
    }
}
