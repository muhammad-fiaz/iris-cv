---
title: "Filters Reference"
description: "API reference for Iris filters — blur, Gaussian, median, bilateral, separable filter, distance transform, filter2D, add weighted, copy, CLAHE, LUT."
keywords: ["filters", "blur", "gaussian blur", "median filter", "bilateral filter", "box blur", "smoothing"]
canonical: "https://muhammad-fiaz.github.io/iris-cv/api/filters"
---

# Filters Reference

Details image filtering and smoothing operation signatures.

::: note
This module is under active development. API signatures may change between versions.
:::

## Blur Filters

```rust
impl<B: Backend> Image<B> {
    pub fn box_blur(self, kernel_size: usize) -> Result<Self>;
    pub fn gaussian_blur(self, kernel_size: usize, sigma: f64) -> Result<Self>;
    pub fn median_blur(self, kernel_size: usize) -> Result<Self>;
    pub fn bilateral_filter(&self, d: isize, sigma_color: f64, sigma_space: f64) -> Result<Self>;
    pub fn sep_filter_2d(&self, kernel_x: &[f32], kernel_y: &[f32]) -> Result<Self>;
}
```

## Utility Filters

```rust
impl<B: Backend> Image<B> {
    pub fn filter2d(
        &self,
        kernel: &[&[f32]],
        anchor: Option<(isize, isize)>,
        delta: f32,
    ) -> Result<Self>;

    pub fn distance_transform(&self) -> Result<Self>;
    pub fn laplacian_of_gaussian(&self, sigma: f64) -> Result<Self>;
    pub fn copy_to(&self, dst: &mut Self, mask: Option<&Self>) -> Result<()>;
}
```

## Blending

```rust
impl<B: Backend> Image<B> {
    pub fn add_weighted(&self, other: &Self, alpha: f32, beta: f32, gamma: f32) -> Result<Self>;
}
```

## Type Conversion

```rust
impl<B: Backend> Image<B> {
    pub fn convert_scale_abs(&self, scale: f32, shift: f32) -> Result<Self>;
}
```
