pub(crate) mod basic;
#[cfg(feature = "lib-coroutine")]
mod coroutine;
#[cfg(feature = "lib-debug")]
mod debug;
#[cfg(feature = "lib-io")]
mod io;
#[cfg(feature = "lib-math")]
pub(crate) mod math;
#[cfg(feature = "lib-os")]
mod os;
#[cfg(feature = "lib-package")]
mod package;
pub(crate) mod stdout;
pub(crate) mod strfmt_num;
#[cfg(feature = "lib-string")]
pub(crate) mod string;
#[cfg(feature = "lib-table")]
pub(crate) mod table;
#[cfg(feature = "lib-utf8")]
mod utf8;
// `pub(crate)` so the VM concat path (`vm::num::write_float`) can share the
// canonical float-stringification (`push_float`) with `tostring`/`print`.
pub(crate) mod util;

pub use basic::load as load_basic;
pub(crate) use basic::{
    ipairs_aux, lua_next, pairs_cont, pcall_cont, return_cont, tostring_cont, xpcall_cont,
};
#[cfg(feature = "lib-coroutine")]
pub(crate) use coroutine::close_cont as coroutine_close_cont;
#[cfg(feature = "lib-coroutine")]
pub use coroutine::load as load_coroutine;
#[cfg(feature = "lib-debug")]
pub use debug::load as load_debug;
#[cfg(feature = "lib-io")]
pub use io::load as load_io;
#[cfg(feature = "lib-math")]
pub use math::load as load_math;
#[cfg(feature = "lib-os")]
pub use os::load as load_os;
#[cfg(feature = "lib-package")]
pub use package::load as load_package;
#[cfg(feature = "lib-string")]
pub use string::load as load_string;
#[cfg(feature = "lib-string")]
pub(crate) use string::{gsub_cont, one_result};
#[cfg(feature = "lib-table")]
pub use table::load as load_table;
#[cfg(feature = "lib-table")]
pub(crate) use table::{sort_cont, sort_len_cont};
#[cfg(feature = "lib-utf8")]
pub use utf8::load as load_utf8;
pub(crate) use util::posrelat;
