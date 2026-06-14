// build.rs — Links the Zig-compiled titankernel static library
// Run `zig build` in the parent directory first to produce libtitankernel.a

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let zig_lib_dir = manifest_dir
        .parent()
        .unwrap()
        .join("zig-out")
        .join("lib");

    // Tell Rust where the Zig static lib lives
    println!("cargo:rustc-link-search=native={}", zig_lib_dir.display());
    println!("cargo:rustc-link-lib=static=titankernel");

    // Rerun if Zig output changes
    println!(
        "cargo:rerun-if-changed={}",
        zig_lib_dir.join("libtitankernel.a").display()
    );
    println!("cargo:rerun-if-changed=../src/ffi_exports.zig");
    println!("cargo:rerun-if-changed=../src/kernels/stream_topk.zig");
    println!("cargo:rerun-if-changed=../src/simd/dispatch.zig");
    println!("cargo:rerun-if-changed=../src/kernels/fused_norm.zig");
}
