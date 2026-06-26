---
title: "Morphology Module Reference"
description: "API reference for Iris morphology module — dilate, erode, morph_open, morph_close, morphology_ex, hit-or-miss, thinning, skeleton extraction, custom kernels, and structuring elements."
keywords: ["morphology", "dilation", "erosion", "opening", "closing", "structuring element", "kernel", "hit-or-miss", "thinning", "skeleton", "Zhang-Suen"]
---

# Morphology Module Reference

Provides morphological image transformations using structuring elements.

## Basic Operations

```rust
impl<B: Backend> Image<B> {
    pub fn dilate(self, kernel_size: usize) -> Result<Self>;
    pub fn erode(self, kernel_size: usize) -> Result<Self>;
    pub fn morph_open(self, kernel_size: usize) -> Result<Self>;
    pub fn morph_close(self, kernel_size: usize) -> Result<Self>;
}
```

## Advanced Operations

### Morphology Ex

```rust
impl<B: Backend> Image<B> {
    pub fn morphology_ex(&self, op: MorphOp, kernel_size: usize) -> Result<Self>;
}
```

Available `MorphOp` variants:

| Variant | Description |
|---|---|
| `MorphOp::Opening` | Erosion followed by dilation. Removes small bright noise. |
| `MorphOp::Closing` | Dilation followed by erosion. Fills small holes. |
| `MorphOp::Gradient` | Difference between dilation and erosion. Highlights boundaries. |
| `MorphOp::TopHat` | Difference between input and opening. Isolates bright features. |
| `MorphOp::BlackHat` | Difference between closing and input. Isolates dark features. |

### Custom Kernel Operations

```rust
impl<B: Backend> Image<B> {
    pub fn dilate_with_kernel(self, kernel: &[&[u8]]) -> Result<Self>;
    pub fn erode_with_kernel(self, kernel: &[&[u8]]) -> Result<Self>;
}
```

## Structuring Elements

```rust
pub struct Morphology;

impl Morphology {
    pub fn get_structuring_element(shape: MorphShape, size: Size<usize>) -> Vec<Vec<u8>>;
}
```

Available `MorphShape` variants:

| Shape | Description |
|---|---|
| `MorphShape::Rect` | Rectangular (flat box) element. |
| `MorphShape::Cross` | Cross-shaped orthogonal lines. |
| `MorphShape::Ellipse` | Elliptical kernel mask. |

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();
let img = Image::<Backend>::open("input.jpg", &device)?;

let dilated = img.clone().dilate(3)?;
let eroded = img.clone().erode(3)?;
let opened = img.clone().morphology_ex(MorphOp::Opening, 3)?;

// Custom cross-shaped kernel
let kernel: Vec<&[u8]> = vec![&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]];
let custom_dilated = img.dilate_with_kernel(&kernel)?;
```

## Hit-or-Miss Transform

```rust
impl<B: Backend> Image<B> {
    pub fn hit_or_miss(&self, pattern: &[&[u8]], bg_pattern: &[&[u8]]) -> Result<Self>;
}
```

Finds pixels matching a specific pattern of foreground (1) and background (0) neighbors. Useful for template matching within binary images.

| Parameter | Description |
|---|---|
| `pattern` | Kernel where 1 marks pixels that must be foreground. |
| `bg_pattern` | Kernel where 1 marks pixels that must be background. |

### Example

```rust
let fg: Vec<&[u8]> = vec![&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]];
let bg: Vec<&[u8]> = vec![&[1, 0, 1], &[0, 0, 0], &[1, 0, 1]];
let hits = img.hit_or_miss(&fg, &bg)?;
```

## Thinning

```rust
impl<B: Backend> Image<B> {
    pub fn thin(&self) -> Result<Self>;
}
```

Applies the Zhang-Suen thinning algorithm to reduce foreground regions to single-pixel-wide strokes while preserving connectivity.

### Example

```rust
let thinned = img.thin()?;
```

## Skeleton Extraction

```rust
impl<B: Backend> Image<B> {
    pub fn skeleton(&self) -> Result<Self>;
}
```

Extracts the topological skeleton of a binary image using iterative morphological thinning. Returns a single-pixel-wide representation of the foreground structure.

### Example

```rust
let skel = img.skeleton()?;
```
