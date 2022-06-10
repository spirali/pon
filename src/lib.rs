pub mod env;
pub mod games;
pub mod process;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;
