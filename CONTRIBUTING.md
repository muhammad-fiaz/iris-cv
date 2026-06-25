# Contributing to Iris

Thank you for your interest in contributing to Iris! As a pure Rust computer vision ecosystem powered by Burn, we welcome contributions of all kinds, including bug fixes, feature requests, new operators, and documentation improvements.

## Code of Conduct

We expect all contributors to adhere to standard respectful collaboration guidelines.

## Development Workflow

1. **Fork and Clone**: Fork the repository on GitHub and clone your fork locally.
2. **Setup Rust**: Make sure you are using the latest stable Rust compiler (Rust 2024 edition is required).
3. **Build the Project**: Run `cargo build` to build the library.
4. **Run Tests**: Before making changes, verify that the existing tests pass with `cargo test`.
5. **Implement Changes**: Add your code changes, and make sure to include unit tests and/or examples if introducing new features.
6. **Lint and Format**:
   - Run `cargo fmt` to auto-format your code.
   - Run `cargo clippy --all-features` to check for lints and warnings. Your code should compile with no warnings.
7. **Commit and Push**: Write clear, descriptive commit messages, and push to your fork.
8. **Submit a Pull Request**: Open a pull request against the `main` branch of the main repository.

## Coding Style

- Follow standard Rust formatting conventions.
- Document public APIs using Rustdoc comments (`///`).
- Keep code clean, modular, and performant. Avoid unnecessary allocations inside tight loops.
- Use `crate::error::{IrisError, Result}` for error handling — never `unwrap()` or `panic!()` in library code.
- All algorithms must be implemented in pure Rust with no external C dependencies.

## Running Examples

Examples require the `wgpu` feature. Run them with:

```bash
cargo run --example image_loading --features wgpu
cargo run --example canny --features wgpu
cargo run --example api_showcase --features wgpu
```

## Module Architecture

The library is organized into these module groups:

| Module | Description |
|--------|-------------|
| `core` | Fundamental types: `Mat`, `Point`, `Rect`, `Scalar`, `Size`, `Rng` |
| `image` | `Image<B>` struct, I/O, geometric transforms, pyramid operations |
| `filters` | Blur (box, Gaussian, median, bilateral), filter2D, distance transform, LoG |
| `edges` | Canny, Sobel, Scharr, Laplacian, Hough lines, Hough circles |
| `threshold` | Binary, Otsu, Triangle, adaptive thresholding |
| `morphology` | Dilate, erode, open, close, hit-or-miss, thinning, skeleton |
| `histogram` | Histogram computation, CLAHE, LUT, 2D histogram, compare |
| `color` | RGB/Gray/HSV/HLS/XYZ/LAB/YUV/YCrCb/CMYK/HSL conversions |
| `contours` | Boundary tracing, convex hull, defects, hierarchy, moments |
| `drawing` | Lines, rectangles, circles, ellipses, polygons, arrows, markers, text |
| `features` | ORB detection, template matching, BFMatcher, FLANN matcher |
| `optical_flow` | Farneback dense flow, Lucas-Kanade sparse flow |
| `tracking` | MOSSE tracker, mean-shift tracker, Kalman filter |
| `segmentation` | Connected components, watershed segmentation |
| `inpaint` | Telea Fast Marching Method inpainting |
| `stereo` | Block matching stereo vision |
| `hog` | Histogram of Oriented Gradients descriptor |
| `photo` | NLM denoising, Mertens exposure fusion |
| `stitching` | Homography-based image stitching |
| `noise` | Gaussian, salt-and-pepper, speckle noise generation |
| `dnn` | ONNX/Safetensors model loading, NMS, blob_from_image |
| `video` | Video I/O, frame iteration, GIF/APNG output |
| `ml` | K-Means clustering |
| `kalman` | Discrete Kalman filter for tracking |
| `aruco` | ArUco marker detection and pose estimation |
| `qr` | QR code detection |
| `barcode` | Barcode detection |
| `face` | Face detection and recognition (DNN-dependent) |
| `object_detection` | Object detection (DNN-dependent) |
| `ocr` | OCR pipeline (DNN-dependent) |
| `camera` | Camera capture and calibration |

## Test Conventions

- Tests use `NdArray` CPU backend via `TestBackend` (no GPU required):
  ```rust
  use crate::test_helpers::{TestBackend, test_device};
  let device = test_device();
  ```
- Create test images with `Tensor::<TestBackend, 3>::from_data(...)` and `Image::new(tensor)`.
- Each new feature must have at least 1 unit test.

## Adding a New Module

1. Create `src/your_module/mod.rs` with the implementation.
2. Add `pub mod your_module;` to `src/lib.rs`.
3. Add re-exports to `src/prelude.rs`.
4. Add tests using `TestBackend`.
5. Create an API doc page at `docs/api/your_module.md`.
6. Add the page to the sidebar in `docs/.vitepress/config.mts`.
7. Add an example in `examples/` if appropriate (with `required-features = ["wgpu"]` in Cargo.toml).
