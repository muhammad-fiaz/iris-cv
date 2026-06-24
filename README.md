<div align="center">

# Iris

<a href="https://muhammad-fiaz.github.io/iris/"><img src="https://img.shields.io/badge/docs-muhammad--fiaz.github.io-blue" alt="Documentation"></a>
<a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-2024_Edition-orange.svg" alt="Rust Version"></a>
<a href="https://github.com/muhammad-fiaz/iris"><img src="https://img.shields.io/github/stars/muhammad-fiaz/iris" alt="GitHub stars"></a>
<a href="https://github.com/muhammad-fiaz/iris/issues"><img src="https://img.shields.io/github/issues/muhammad-fiaz/iris" alt="GitHub issues"></a>
<a href="https://github.com/muhammad-fiaz/iris/pulls"><img src="https://img.shields.io/github/issues-pr/muhammad-fiaz/iris" alt="GitHub pull requests"></a>
<a href="https://github.com/muhammad-fiaz/iris"><img src="https://img.shields.io/github/last-commit/muhammad-fiaz/iris" alt="GitHub last commit"></a>
<a href="https://github.com/muhammad-fiaz/iris"><img src="https://img.shields.io/github/license/muhammad-fiaz/iris" alt="License"></a>
<a href="https://github.com/muhammad-fiaz/iris/actions/workflows/ci.yml"><img src="https://github.com/muhammad-fiaz/iris/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/badge/platforms-linux%20%7C%20windows%20%7C%20macos-blue" alt="Supported Platforms">
<a href="https://github.com/muhammad-fiaz/iris/releases/latest"><img src="https://img.shields.io/github/v/release/muhammad-fiaz/iris?label=Latest%20Release&style=flat-square" alt="Latest Release"></a>
<a href="https://pay.muhammadfiaz.com"><img src="https://img.shields.io/badge/Sponsor-pay.muhammadfiaz.com-ff69b4?style=flat&logo=heart" alt="Sponsor"></a>
<a href="https://github.com/sponsors/muhammad-fiaz"><img src="https://img.shields.io/badge/Sponsor-💖-pink?style=social&logo=github" alt="GitHub Sponsors"></a>
<a href="https://hits.sh/muhammad-fiaz/iris/"><img src="https://hits.sh/muhammad-fiaz/iris.svg?label=Visitors&extraCount=0&color=green" alt="Repo Visitors"></a>

<p><em>A fast computer vision library framework in Rust.</em></p>

<b><a href="https://muhammad-fiaz.github.io/iris/">Documentation</a> |
<a href="https://muhammad-fiaz.github.io/iris/api">API Reference</a> |
<a href="CONTRIBUTING.md">Contributing</a></b>

</div>

A Rust-powered, cross-platform computer vision and deep learning library designed with a clean, modular, library-first architecture. Completely native with zero external C dependencies.

**If you love `Iris`, make sure to give it a star!**

---

<details>
<summary><strong>Table of Contents</strong> (click to expand)</summary>

- [Prerequisites & Supported Platforms](#prerequisites--supported-platforms)
- [Features of Iris](#features-of-iris)
- [Installation](#installation)
  - [Build from Source](#build-from-source)
- [Library Usage](#library-usage)
- [Examples](#examples)
- [License](#license)

</details>

---

<details>
<summary><strong>Features of Iris</strong> (click to expand)</summary>

| Feature | Description |
|---------|-------------|
| **Image Representation** | Custom `Image<B>` wrapping a Burn `Tensor<B, 3>` for high performance. |
| **Image I/O** | Native load and save support for popular formats (PNG, JPEG, etc.). |
| **Color Conversions** | Fast transformations between RGB, Grayscale, and other color models. |
| **Geometric Operations** | Warping, crop, flip, rotate, scale, affine/perspective warps, and remapping. |
| **Filtering & Blur** | Smooth and blur operations (box, Gaussian, median, bilateral, separable). |
| **Edge Detection** | Canny, Sobel, Laplacian, and custom gradient filters. |
| **Thresholding** | Binary, inverse, trunc, Otsu's, and adaptive thresholding. |
| **Morphological Operations** | Dilation, erosion, opening, closing, gradient, top hat, black hat. |
| **Contours & Shapes** | Suzuki-Abe boundary tracking, convex hull, moments, bounding boxes, polygon approximation. |
| **Drawing Canvas** | Lines, rectangles, circles, polygons, and custom text rendering using a built-in bitmap font. |
| **DNN Inference** | Native ONNX, Safetensors, and Burn weight loading, NMS, and implicit `pretrained()` model loaders. |
| **ArUco & QR Detection** | Marker tracking, pose estimation, and QR/barcode reader pipelines. |
| **Optical Flow & Tracking** | Dense and sparse tracking pipelines with background subtraction. |
| **WGPU & GPU Acceleration** | Native acceleration across CUDA, Vulkan, Metal, and WGPU through Burn's backend abstraction. |

</details>

---

<details>
<summary><strong>Prerequisites & Supported Platforms</strong> (click to expand)</summary>

## Prerequisites

Before installing Iris, ensure you have:

- **Rust**: v1.85.0+ (Rust 2024 Edition). Install via [rustup](https://rustup.rs/):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Cargo**: Comes bundled with Rust. Verify with `cargo --version`.

## Supported Platforms

Iris supports a wide range of platforms and architectures:
- **Windows 10+ / 11+**
- **Linux** (Vulkan/CUDA acceleration support)
- **macOS** (Metal acceleration support)

</details>

---

## Installation

### Build from Source
```bash
git clone https://github.com/muhammad-fiaz/iris.git
cd iris
cargo build --release
```

---

## Cargo Features

`Iris` provides several features to customize compilation and backend acceleration:

| Feature | Description | Enabled by Default |
|---|---|---|
| `parallel` | Enables CPU multi-threading parallelization using `rayon` for all CPU pixel-processing operators (filters, gradients, morphology, logical/bitwise ops, thresholding, warping). | **Yes** |
| `wgpu` | Enables the WGPU backend support for hardware-accelerated deep learning via the Burn framework. | **Yes** |
| `safetensors` | Enables native loading of model weights in `.safetensors` format. | **Yes** |
| `gpui` | Enables GPU-accelerated desktop UI window rendering using Zed's `gpui` and `gpui-component` frameworks (cross-platform on Linux, macOS, and Windows). | No |
| `cuda` | Enables CUDA acceleration for the Burn backend. | No |
| `tch` | Enables LibTorch backend acceleration. | No |
| `ndarray` | Enables lightweight, pure CPU ndarray execution fallback. | No |

---

## Library Usage

To use `iris` in your Rust project, run:

```bash
cargo add iris
```

Or add it directly to your `Cargo.toml` under dependencies:

```toml
[dependencies]
iris = "0.0.0"
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



In your Rust code:

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define Burn device & backend type
    let device = Default::default();
    
    // 2. Open an image
    let img: Image<Wgpu> = Image::open("test.png", &device)?;
    println!("Loaded image with shape: {:?}", img.shape());

    // 3. Convert to grayscale and apply Canny edges
    let gray = img.grayscale()?;
    let edges = gray.canny(50.0, 150.0)?;

    // 4. Draw bounding box and text
    let mut canvas = edges.to_rgb()?;
    canvas = canvas.draw_rectangle(
        Rect::new(10, 10, 100, 100),
        Scalar::new(1.0, 0.0, 0.0, 1.0),
        2
    )?;
    canvas.save("output.png")?;

    Ok(())
}
```

---

## Examples

Run any example to see Iris in action:

```bash
cargo run --example image_loading
cargo run --example canny
cargo run --example contours
cargo run --example qr_detection
cargo run --example face_recognition
cargo run --example onnx_inference
cargo run --example gui_windows
cargo run --example safetensors_loading
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
