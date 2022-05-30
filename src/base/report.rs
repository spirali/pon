use crate::base::process::Process;
use serde::Serialize;

#[derive(Serialize)]
pub struct RunReport<ProcessT: Process> {
    pub network: String,
    pub steps: usize,
    pub process: ProcessT::ReportT,
}
