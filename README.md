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
<a href="https://hits.sh/github.com/muhammad-fiaz/iris/"><img alt="Hits" src="https://hits.sh/github.com/muhammad-fiaz/iris.svg"/></a>

<p><em>A fast computer vision library framework in Rust.</em></p>

<b><a href="https://muhammad-fiaz.github.io/iris/">Documentation</a> |
<a href="https://muhammad-fiaz.github.io/iris/api">API Reference</a> |
<a href="CONTRIBUTING.md">Contributing</a></b>

</div>

A Rust-powered, cross-platform computer vision and deep learning library designed with a clean, modular, library-first architecture. Completely native with zero external C dependencies.

**If you love `Iris`, make sure to give it a star!**

> [!NOTE]
> This project is still in active development. APIs and features may change before the first stable release.

---

<details>
<summary><strong>Table of Contents</strong> (click to expand)</summary>

- [Prerequisites & Supported Platforms](#prerequisites--supported-platforms)
- [Features of Iris](#features-of-iris)
- [Installation](#installation)
  - [Build from Source](#build-from-source)
- [Library Usage](#library-usage)
- [Examples](#examples)
- [Cargo Features](#cargo-features)
- [License](#license)

</details>

---

<details>
<summary><strong>Features of Iris</strong> (click to expand)</summary>

| Feature | Description |
|---------|-------------|
| **Image Representation** | Custom `Image<B>` wrapping a Burn `Tensor<B, 3>` for high performance. |
| **Image I/O** | Native load and save support for PNG, JPEG, GIF, QOI, ICO, BMP, TIFF, WebP, APNG. |
| **Color Conversions** | RGB, Grayscale, HSV, HLS, XYZ, LAB, YUV, YCrCb conversions with channel split/merge. |
| **Geometric Operations** | Warp, crop, flip, rotate, scale, affine/perspective transforms, remapping, resize. |
| **Filtering & Blur** | Box, Gaussian, median, bilateral, separable, custom kernel convolution (filter2D). |
| **Edge Detection** | Canny, Sobel, Scharr, Laplacian, LoG (Laplacian of Gaussian). |
| **Thresholding** | Binary, Otsu's, Triangle, adaptive (mean/Gaussian) thresholding. |
| **Histogram** | Histogram equalization, CLAHE, apply LUT, compare histograms (4 methods), per-channel operations. |
| **Morphological Operations** | Dilation, erosion, opening, closing, gradient, top/bottom hat, custom kernels. |
| **Contours & Shapes** | Suzuki-Abe boundary tracking, convex hull, moments, bounding boxes, polygon approximation, distance transform. |
| **Drawing Canvas** | Lines, rectangles, circles, ellipses, polygons, arrows, markers, text rendering with bitmap font. |
| **Noise Generation** | Gaussian, salt-and-pepper, and speckle noise with custom parameters. |
| **Feature Detection** | ORB feature detection (FAST corners + BRIEF descriptors), template matching (6 methods). |
| **Dense Optical Flow** | Farneback multi-scale Gaussian pyramid flow estimation. |
| **Sparse Optical Flow** | Lucas-Kanade pyramidal feature tracking. |
| **Object Tracking** | MOSSE correlation filter tracker. |
| **Video Module** | Read/write video files, frame iteration, GIF/APNG/JPEG/QOI/PNG output, metadata extraction. |
| **DNN Inference** | Native ONNX, Safetensors, and Burn weight loading, NMS. |
| **ArUco & QR Detection** | Marker tracking, pose estimation, and QR/barcode reader pipelines. |
| **Image Utilities** | addWeighted blending, convert_scale_abs, copy_to with mask, normalize, in_range masking. |
| **WGPU & GPU Acceleration** | Native acceleration across CUDA, Vulkan, Metal, and WGPU through Burn's backend. |
| **Parallel Processing** | Rayon-powered parallelism across filters, gradients, morphology, thresholding, and warping. |

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
use burn::backend::wgpu::{Wgpu, WgpuDevice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    // 1. Open an image
    let img: Image<Backend> = Image::open("assets/images/gradient.png", &device)?;
    println!("Loaded image with shape: {:?}", img.shape());

    // 2. Convert to grayscale and apply Canny edges
    let gray = img.grayscale()?;
    let edges = gray.canny(50.0, 150.0)?;

    // 3. Draw bounding box and save
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

Run any example to see Iris in action. All examples require the `wgpu` feature:

```bash
cargo run --example image_loading --features wgpu
cargo run --example canny --features wgpu
cargo run --example filters --features wgpu
cargo run --example drawing --features wgpu
cargo run --example color_processing --features wgpu
cargo run --example image_utils --features wgpu
cargo run --example contours --features wgpu
cargo run --example morphology --features wgpu
cargo run --example threshold --features wgpu
cargo run --example optical_flow --features wgpu
cargo run --example tracking --features wgpu
cargo run --example qr_detection --features wgpu
cargo run --example face_recognition --features wgpu
cargo run --example onnx_inference --features wgpu
cargo run --example photo_processing --features wgpu
cargo run --example segmentation --features wgpu
cargo run --example kmeans_clustering --features wgpu
```

---

## Cargo Features

`Iris` provides several features to customize compilation and backend acceleration:

| Feature | Description | Enabled by Default |
|---|---|---|
| `ndarray` | Lightweight, pure CPU ndarray execution backend (used for tests). | **Yes** |
| `safetensors` | Enables native loading of model weights in `.safetensors` format. | **Yes** |
| `wgpu` | Enables the WGPU backend support for hardware-accelerated deep learning via Burn. | No |
| `gpui` | Enables GPU-accelerated desktop UI window rendering using Zed's GPUI framework. | No |
| `cuda` | Enables CUDA acceleration for the Burn backend. | No |
| `tch` | Enables LibTorch backend acceleration. | No |

> **Note**: `rayon` is a required dependency for parallel processing — it is always included.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
