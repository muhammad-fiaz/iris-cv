---
title: "Image Operators Reference"
description: "API reference for Iris image operators — conversions, I/O, geometric transforms, resize, crop, flip, rotate, warp, remap, lens undistortion, Gaussian pyramid, and morphological operations."
keywords: ["image operators", "conversions", "grayscale", "RGB", "geometric transforms", "resize", "crop", "flip", "undistort", "lens undistortion", "pyr_down", "pyr_up", "Gaussian pyramid"]
canonical: "https://muhammad-fiaz.github.io/iris-cv/api/image"
---

# Image Operators Reference

Exposes image allocation, transformation, and pixel operators on the `Image` struct.

::: note
This module is under active development. API signatures may change between versions.
:::

## Image Struct

```rust
pub struct Image<B: Backend> {
    pub tensor: Tensor<B, 3>,
}

impl<B: Backend> Image<B> {
    pub fn new(tensor: Tensor<B, 3>) -> Self;
    pub fn shape(&self) -> [usize; 3];
    pub fn channels(&self) -> usize;
    pub fn height(&self) -> usize;
    pub fn width(&self) -> usize;
}
```

## Conversions

```rust
impl<B: Backend> Image<B> {
    pub fn grayscale(&self) -> Result<Self>;
    pub fn to_rgb(&self) -> Result<Self>;
}
```

## Image I/O

```rust
impl<B: Backend> Image<B> {
    pub fn open(path: impl AsRef<Path>, device: &B::Device) -> Result<Self>;
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()>;
}
```

## Geometric Transforms

```rust
impl<B: Backend> Image<B> {
    pub fn resize(&self, new_width: usize, new_height: usize) -> Result<Self>;
    pub fn crop(&self, x: usize, y: usize, width: usize, height: usize) -> Result<Self>;
    pub fn flip(&self, horizontal: bool, vertical: bool) -> Result<Self>;
    pub fn rotate(&self, angle_degrees: u32) -> Result<Self>;
    pub fn transpose(&self) -> Result<Self>;
    pub fn warp_affine(&self, m: [[f64; 3]; 2], new_width: usize, new_height: usize) -> Result<Self>;
    pub fn warp_perspective(&self, m: [[f64; 3]; 3], new_width: usize, new_height: usize) -> Result<Self>;
    pub fn remap(&self, map_x: &Tensor<B, 2>, map_y: &Tensor<B, 2>) -> Result<Self>;
}
```

## Additional Operations

```rust
impl<B: Backend> Image<B> {
    pub fn gaussian_pyramid(&self, levels: usize) -> Result<Vec<Self>>;
    pub fn integral_image(&self) -> Result<Self>;
    pub fn flood_fill(&self, seed_x: usize, seed_y: usize, fill_value: f32, lo_diff: f32, hi_diff: f32) -> Result<Self>;
}
```

## Lens Undistortion

```rust
impl<B: Backend> Image<B> {
    pub fn undistort(&self, camera_matrix: &Tensor<B, 2>, dist_coeffs: &[f32]) -> Result<Self>;
}
```

Removes lens distortion using a pinhole camera model with radial and tangential distortion coefficients.

| Parameter | Description |
|---|---|
| `camera_matrix` | 3×3 intrinsic camera matrix (focal length, principal point). |
| `dist_coeffs` | Distortion coefficients `[k1, k2, p1, p2, k3, ...]`. |

## Gaussian Pyramid

```rust
impl<B: Backend> Image<B> {
    pub fn pyr_down(&self) -> Result<Self>;
    pub fn pyr_up(&self) -> Result<Self>;
}
```

Applies Gaussian pyramid operations for multi-scale image analysis.

| Method | Description |
|---|---|
| `pyr_down()` | Reduces image dimensions by half (blurred + downsampled). |
| `pyr_up()` | Doubles image dimensions (upsampled + blurred). |

### Example

```rust
let small = img.pyr_down()?;
let restored = small.pyr_up()?;
```
