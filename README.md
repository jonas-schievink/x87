# x87 FPU emulator

[![crates.io](https://img.shields.io/crates/v/x87.svg)](https://crates.io/crates/x87)
[![docs.rs](https://docs.rs/x87/badge.svg)](https://docs.rs/x87/)
[![Build Status](https://travis-ci.org/jonas-schievink/x87.svg?branch=master)](https://travis-ci.org/jonas-schievink/x87)

This crate is a software implementation of the x87 family of floating-point
coprocessors from the 80s that are still in use in today's x86-64 CPUs. It aims
to be correct and indistinguishable from a "real" x87 FPU. Reaching this goal
will still take a *lot* of work, though.

If you feel like helping out, don't hesitate to open issues or direct PRs, but
note that the code is very messy since it's my first venture into floating-point
implementations.

Please refer to the [changelog](CHANGELOG.md) to see what changed in the last
releases.

## Usage

Start by adding an entry to your `Cargo.toml`:

```toml
[dependencies]
x87 = "0.1.0"
```

Then import the crate into your Rust code:

```rust
extern crate x87;
```
