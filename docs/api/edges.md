---
title: "Edges Module Reference"
description: "API reference for Iris edges module — Sobel, Canny, Scharr, Laplacian edge detection, Hough lines, and Hough circles."
keywords: ["edges", "edge detection", "Canny", "Sobel", "Scharr", "Laplacian", "Hough lines", "Hough circles", "gradient"]
---

# Edges Module Reference

Provides gradient computation, edge detection, and geometric shape detection.

## Edge Detection

```rust
impl<B: Backend> Image<B> {
    pub fn sobel(&self) -> Result<Self>;
    pub fn scharr(&self) -> Result<(Self, Self)>;
    pub fn laplacian(&self, kernel_size: usize) -> Result<Self>;
    pub fn canny(&self, low_threshold: f32, high_threshold: f32) -> Result<Self>;
}
```

## Hough Transforms

### HoughLinesP

Probabilistic Hough Line Transform detecting line segments in a binary edge image.

```rust
pub type LineSegment = ((usize, usize), (usize, usize));

impl<B: Backend> Image<B> {
    pub fn hough_lines_p(
        &self,
        rho: f32,
        theta: f32,
        threshold: u32,
        min_line_length: u32,
        max_line_gap: u32,
    ) -> Result<Vec<LineSegment>>;
}
```

### HoughCircles

Hough Circle Transform using gradient information.

```rust
impl<B: Backend> Image<B> {
    pub fn hough_circles(
        &self,
        dp: f32,
        min_dist: f32,
        param1: f32,
        param2: f32,
        min_radius: usize,
        max_radius: usize,
    ) -> Result<Vec<(usize, usize, usize)>>;
}
```

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();
let img = Image::<Backend>::open("input.jpg", &device)?;

// Edge detection
let edges = img.canny(0.1, 0.3)?;

// Hough line detection
let lines = edges.hough_lines_p(1.0, std::f32::consts::PI / 180.0, 50, 50, 10)?;
```
