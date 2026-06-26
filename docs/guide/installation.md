---
title: "Installation"
description: "Install Iris in your Rust project — cargo add, Git dependency, feature flags for wgpu, ndarray, safetensors, and gpui."
keywords: ["installation", "cargo add", "dependencies", "feature flags", "wgpu", "setup"]
---

# Installation

Iris compiles with a standard Rust toolchain. We recommend using the latest stable release of Rust (v1.85.0+ or Rust 2024 edition).

To install Iris in your Rust project, run:

```bash
cargo add iris-cv
```

### Development Version (Git)

To use the latest development branch directly from GitHub, run:

```bash
cargo add iris-cv --git https://github.com/muhammad-fiaz/iris-cv
```

Or add the following to your `Cargo.toml`:

```toml
[dependencies]
iris-cv = { git = "https://github.com/muhammad-fiaz/iris-cv" }
```

## Cargo Features

You can customize Iris by specifying feature flags when running `cargo add`:

```bash
cargo add iris-cv --no-default-features --features wgpu
```

Or configure features directly inside your `Cargo.toml` under dependencies:

```toml
[dependencies]
iris-cv = { version = "0.0.0", default-features = false, features = ["wgpu"] }
```

### Default Features

- **`ndarray`** (Enabled by default): Includes the CPU ndarray backend for testing and environments without GPU.
- **`safetensors`** (Enabled by default): Includes support for loading `.safetensors` model weights for DNN inference.

### Optional Features

- **`wgpu`**: Enables the Burn Wgpu backend, providing hardware-accelerated execution on graphics hardware. Required for all examples.
- **`gpui`**: Integrates Zed's `gpui` and `gpui-component` libraries for GPU-accelerated windows and UI layout.

### Running Examples

```bash
git clone https://github.com/muhammad-fiaz/iris-cv.git
cd iris-cv
cargo run --example canny --features wgpu
```
