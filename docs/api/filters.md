---
title: "Filters Reference"
description: "API reference for Iris filters — box, Gaussian, median, bilateral blur, separable filter, distance transform, filter2D, add weighted, and copy."
keywords: ["filters", "blur", "gaussian blur", "median filter", "bilateral filter", "box blur", "smoothing"]
---

# Filters Reference

Details image filtering and smoothing operation signatures.

## Blur Filters

```rust
impl<B: Backend> Image<B> {
    pub fn box_blur(self, kernel_size: usize) -> Result<Self>;
    pub fn gaussian_blur(self, kernel_size: usize, sigma: f64) -> Result<Self>;
    pub fn median_blur(self, kernel_size: usize) -> Result<Self>;
    pub fn bilateral_filter(self, d: usize, sigma_color: f32, sigma_space: f64) -> Result<Self>;
    pub fn sep_filter_2d(&self, kernel_x: &[f32], kernel_y: &[f32]) -> Result<Self>;
}
```

## Utility Filters

```rust
impl<B: Backend> Image<B> {
    pub fn filter2d(&self, kernel: &[Vec<f32>]) -> Result<Self>;
    pub fn distance_transform(&self) -> Result<Self>;
    pub fn laplacian_of_gaussian(&self, sigma: f64) -> Result<Self>;
    pub fn copy_to(&self, mask: &Image<B>) -> Result<Self>;
}
```

## Blending

```rust
impl<B: Backend> Image<B> {
    pub fn add_weighted(
        src1: &Image<B>,
        alpha: f32,
        src2: &Image<B>,
        beta: f32,
        gamma: f32,
    ) -> Result<Image<B>>;
}
```

## Type Conversion

```rust
impl<B: Backend> Image<B> {
    pub fn convert_scale_abs(&self) -> Result<Self>;
}
```
