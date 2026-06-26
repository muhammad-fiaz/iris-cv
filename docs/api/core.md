---
title: "Core Module Reference"
description: "API reference for Iris core module — Point, Rect, Size, Scalar, Mat, Rng, image math ops, bitwise ops, normalization, and statistics."
keywords: ["core module", "Point", "Rect", "Size", "Scalar", "Mat", "Rng", "geometry types"]
---

# Core Module Reference

The `core` module contains common algebraic representations, basic geometries, utility structures, and image math operations.

## Geometries

### `Point<T>`

```rust
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point<T> {
    pub fn new(x: T, y: T) -> Self;
}
```

### `Rect<T>`

```rust
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

impl<T> Rect<T> {
    pub fn new(x: T, y: T, width: T, height: T) -> Self;
}
```

### `Size<T>`

```rust
pub struct Size<T> {
    pub width: T,
    pub height: T,
}

impl<T> Size<T> {
    pub fn new(width: T, height: T) -> Self;
}
```

### `Scalar`

```rust
pub struct Scalar(pub [f64; 4]);

impl Scalar {
    pub fn new(v0: f64, v1: f64, v2: f64, v3: f64) -> Self;
    pub fn all(val: f64) -> Self;
}
```

### `Rng`

Pseudo-random number generator for reproducible random operations.

```rust
pub struct Rng { /* ... */ }
impl Rng {
    pub fn new(seed: u64) -> Self;
    pub fn next_u64(&mut self) -> u64;
    pub fn next_u32(&mut self) -> u32;
    pub fn next_f32(&mut self) -> f32;
    pub fn next_f64(&mut self) -> f64;
}
```

## Matrix Representation

### `Mat<B, const D: usize>`

```rust
pub struct Mat<B: Backend, const D: usize> {
    pub tensor: Tensor<B, D>,
}

impl<B: Backend, const D: usize> Mat<B, D> {
    pub fn new(tensor: Tensor<B, D>) -> Self;
    pub fn shape(&self) -> Vec<usize>;
    pub fn total(&self) -> usize;
}
```

## Image Math Operations

All operations are performed element-wise between two images.

```rust
impl<B: Backend> Image<B> {
    pub fn add(&self, other: &Self) -> Result<Self>;
    pub fn subtract(&self, other: &Self) -> Result<Self>;
    pub fn multiply(&self, other: &Self) -> Result<Self>;
    pub fn divide(&self, other: &Self) -> Result<Self>;
    pub fn absdiff(&self, other: &Self) -> Result<Self>;
}
```

## Bitwise Operations

```rust
impl<B: Backend> Image<B> {
    pub fn bitwise_and(&self, other: &Self) -> Result<Self>;
    pub fn bitwise_or(&self, other: &Self) -> Result<Self>;
    pub fn bitwise_xor(&self, other: &Self) -> Result<Self>;
    pub fn bitwise_not(&self) -> Result<Self>;
}
```

## In-Range

Checks which pixels fall within a color range and creates a binary mask.

```rust
impl<B: Backend> Image<B> {
    pub fn in_range(&self, low: &[f32], high: &[f32]) -> Result<Self>;
}
```

## Statistics

```rust
impl<B: Backend> Image<B> {
    pub fn normalize(&self, min_val: f32, max_val: f32) -> Result<Self>;
    pub fn mean(&self) -> Result<Vec<f64>>;
    pub fn mean_std_dev(&self) -> Result<(Vec<f64>, Vec<f64>)>;
    pub fn min_max_loc(&self) -> Result<(f64, f64, Point<usize>, Point<usize>)>;
    pub fn count_non_zero(&self) -> Result<usize>;
}
```
