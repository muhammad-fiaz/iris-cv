---
title: "Installation"
description: "Install Iris in your Rust project — cargo add, Git dependency, feature flags for wgpu, parallel, cuda, tch, ndarray, and gpui."
keywords: ["installation", "cargo add", "dependencies", "feature flags", "wgpu", "cuda", "setup"]
---

# Installation

Iris compiles with a standard Rust toolchain. We recommend using the latest stable release of Rust (v1.85.0+ or Rust 2024 edition).

To install Iris in your Rust project, run:

```bash
cargo add iris
```

### Development Version (Git)

To use the latest development branch directly from GitHub, run:

```bash
cargo add iris --git https://github.com/muhammad-fiaz/iris
```

Or add the following to your `Cargo.toml`:

```toml
[dependencies]
iris = { git = "https://github.com/muhammad-fiaz/iris" }
```


## Cargo Features

You can customize Iris by specifying feature flags when running `cargo add`:

```bash
cargo add iris --no-default-features --features wgpu,parallel
```

Or configure features directly inside your `Cargo.toml` under dependencies:

```toml
[dependencies]
iris = { version = "0.0.0", default-features = false, features = ["wgpu", "parallel"] }
```


### Core Features

- **`parallel`** (Enabled by default): Enables CPU multi-threading parallelization using `rayon`. All major filters, morphological operations, logical/bitwise operations, and warping transforms use multi-core scheduling.
- **`safetensors`** (Enabled by default): Includes support for loading `.safetensors` model weights for DNN inference.
- **`wgpu`** (Enabled by default): Enables the Burn Wgpu backend, providing hardware-accelerated execution on graphics hardware.

### Extra Features

- **`gpui`**: Integrates Zed's `gpui` and `gpui-component` libraries inside the direct `gui::Gui` window manager. Allows GPU-accelerated windows and UI layout loops across Windows, Linux, and macOS.
- **`cuda`**: Enables CUDA acceleration support for deep learning inference.
- **`tch`**: Enables LibTorch backend integration.
- **`ndarray`**: Enables a lightweight, pure CPU ndarray backend, useful for embedded systems or serverless deployments with small memory footprints.
