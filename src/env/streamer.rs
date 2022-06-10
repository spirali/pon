use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::thread::JoinHandle;

pub struct Streamer {
    handle: JoinHandle<()>,
    sender: std::sync::mpsc::SyncSender<String>,
}

impl Streamer {
    pub fn new(path: &Path) -> std::io::Result<Self> {
        let mut output = BufWriter::new(File::create(path)?);
        let (sender, recv) =
            std::sync::mpsc::sync_channel::<String>(2 * rayon::current_num_threads());
        let handle = std::thread::spawn(move || {
            for data in recv.into_iter() {
                writeln!(output, "{}", data).unwrap();
            }
        });
        Ok(Self { handle, sender })
    }
    pub fn send<T: Serialize>(&self, value: &T) {
        let serialized_value: String = serde_json::to_string(value).unwrap();
        self.sender.send(serialized_value).unwrap();
    }

    pub fn join(self) {
        drop(self.sender);
        self.handle.join().unwrap()
    }
}

/*pub fn run_experiment<C: Sync, T: Serialize>(
    networks: impl IntoParallelIterator<Item = Network>,
    conf_iter: impl ParallelIterator<Item = C> + Clone + Sync,
    task_fn: impl Fn(&Network, &C) -> T + Send + Sync,
    replications: usize,
    output_path: &Path,
) {
    let mut output = BufWriter::new(File::create(output_path).unwrap());

    let (send, recv) = std::sync::mpsc::sync_channel::<String>(2 * rayon::current_num_threads());

    let write_thread = std::thread::spawn(move || {
        for data in recv.into_iter() {
            writeln!(output, "{}", data).unwrap();
        }
    });

    networks.into_par_iter().for_each(|network| {
        conf_iter.clone().into_par_iter().for_each(|conf| {
            (0..replications).into_par_iter().for_each(|_| {
                let value = task_fn(&network, &conf);
                let serialized_value: String = serde_json::to_string(&value).unwrap();
                send.send(serialized_value).unwrap();
            })
        });
    });

    drop(send);

    /*    confs.into_par_iter().for_each(|value| {

    */
    dbg!("JOIN");
    write_thread.join().unwrap();
}
*/
