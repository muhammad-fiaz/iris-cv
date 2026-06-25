---
title: "Image Operators Reference"
description: "API reference for Iris image operators — conversions, I/O, geometric transforms, resize, crop, flip, rotate, warp, remap, lens undistortion, Gaussian pyramid, and morphological operations."
keywords: ["image operators", "conversions", "grayscale", "RGB", "geometric transforms", "resize", "crop", "flip", "undistort", "lens undistortion", "pyr_down", "pyr_up", "Gaussian pyramid"]
---

# Image Operators Reference

Exposes image allocation, transformation, and pixel operators on the `Image` struct.

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
    pub fn crop(&self, rect: Rect<usize>) -> Result<Self>;
    pub fn flip(&self, axis: i32) -> Result<Self>;
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
    pub fn gaussian_pyramid(&self) -> Result<Self>;
    pub fn integral_image(&self) -> Result<Self>;
    pub fn flood_fill(&self, seed: Point<usize>, color: Scalar) -> Result<Self>;
}
```

## Lens Undistortion

```rust
impl<B: Backend> Image<B> {
    pub fn undistort(&self, camera_matrix: [[f64; 3]; 3], dist_coeffs: &[f64; 5]) -> Result<Self>;
}
```

Removes lens distortion using a pinhole camera model with radial and tangential distortion coefficients.

| Parameter | Description |
|---|---|
| `camera_matrix` | 3×3 intrinsic camera matrix (focal length, principal point). |
| `dist_coeffs` | Five distortion coefficients `[k1, k2, p1, p2, k3]`. |

### Example

```rust,ignore
let camera = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]];
let dist = [-0.1, 0.01, 0.001, -0.002, 0.0];
let corrected = img.undistort(camera, &dist)?;
```

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

```rust,ignore
let small = img.pyr_down()?;
let restored = small.pyr_up()?;
```
