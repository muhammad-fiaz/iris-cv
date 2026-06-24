---
title: "Image Operators Reference"
description: "API reference for Iris image operators — conversions, I/O, arithmetic, geometric transforms, resize, warp, rotate, and remap."
keywords: ["image operators", "conversions", "grayscale", "RGB", "arithmetic", "geometric transforms", "resize"]
---

# Image Operators Reference

Exposes image allocation, transformation, and pixel arithmetic operators on the `Image` struct.

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
    /// Converts a 3-channel image to a single-channel grayscale image.
    pub fn grayscale(&self) -> Result<Self>;

    /// Converts a single-channel grayscale image to a 3-channel RGB image.
    pub fn to_rgb(&self) -> Result<Self>;
}
```

## Image I/O

```rust
pub fn load_image<B: Backend>(path: impl AsRef<Path>, device: &B::Device) -> Result<Image<B>>;
pub fn save_image<B: Backend>(image: &Image<B>, path: impl AsRef<Path>) -> Result<()>;
```

## Arithmetic Operators

```rust
impl<B: Backend> Image<B> {
    pub fn add(&self, other: &Self) -> Result<Self>;
    pub fn subtract(&self, other: &Self) -> Result<Self>;
    pub fn multiply(&self, other: &Self) -> Result<Self>;
    pub fn divide(&self, other: &Self) -> Result<Self>;
    pub fn absdiff(&self, other: &Self) -> Result<Self>;
}
```

## Geometric Transformations

```rust
impl<B: Backend> Image<B> {
    pub fn resize(&self, new_width: usize, new_height: usize) -> Result<Self>;
    pub fn warp_affine(&self, m: [[f64; 3]; 2], new_width: usize, new_height: usize) -> Result<Self>;
    pub fn warp_perspective(&self, m: [[f64; 3]; 3], new_width: usize, new_height: usize) -> Result<Self>;
    pub fn remap(&self, map_x: &Tensor<B, 2>, map_y: &Tensor<B, 2>) -> Result<Self>;
    pub fn rotate(&self, angle_degrees: u32) -> Result<Self>;
}
```
