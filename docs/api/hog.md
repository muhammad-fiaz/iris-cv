---
title: "HOG Descriptor Reference"
description: "API reference for Iris HOG descriptor — Histogram of Oriented Gradients for feature extraction."
keywords: ["HOG", "histogram of oriented gradients", "feature extraction", "descriptor"]
canonical: "https://muhammad-fiaz.github.io/iris-cv/api/hog"
---

# HOG Descriptor Reference

Computes Histogram of Oriented Gradients (HOG) descriptors for image regions.

::: note
This module is under active development. API signatures may change between versions.
:::

## HogDescriptor

```rust
pub struct HogDescriptor {
    cell_size: usize,
    block_size: usize,
    nbins: usize,
}

impl HogDescriptor {
    pub fn new(cell_size: usize, block_size: usize, nbins: usize) -> Self;
    pub fn compute<B: Backend>(&self, image: &Image<B>) -> Result<Tensor<B, 1>>;
}
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `cell_size` | `usize` | Size of each cell in pixels (e.g., 8). |
| `block_size` | `usize` | Number of cells per block (e.g., 2 for 2×2). |
| `nbins` | `usize` | Number of orientation bins (e.g., 9). |

### Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();

let img = Image::<Backend>::open("person.png", &device)?;
let hog = HogDescriptor::new(8, 2, 9);
let descriptor = hog.compute(&img)?;
println!("HOG descriptor shape: {:?}", descriptor.dims());
```
