---
title: "HOG Descriptor Reference"
description: "API reference for Iris HOG descriptor — Histogram of Oriented Gradients for feature extraction."
keywords: ["HOG", "histogram of oriented gradients", "feature extraction", "pedestrian detection"]
canonical: "https://muhammad-fiaz.github.io/iris/api/hog"
---

# HOG Descriptor Reference

Computes Histogram of Oriented Gradients descriptors for feature extraction and object detection.

::: note
This module is under active development. API signatures may change between versions.
:::

## HogDescriptor

```rust
pub struct HogDescriptor {
    cell_size: (u32, u32),
    block_size: (u32, u32),
    nbins: u32,
    block_stride: (u32, u32),
}

impl HogDescriptor {
    pub fn new(
        cell_size: (u32, u32),
        block_size: (u32, u32),
        nbins: u32,
    ) -> Self;

    pub fn compute<B: Backend>(
        &self,
        image: &Image<B>,
        device: &B::Device,
    ) -> Result<Vec<f32>>;
}
```

### Constructor

#### `new(cell_size, block_size, nbins)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `cell_size` | `(u32, u32)` | Size of each cell in pixels `(width, height)`. |
| `block_size` | `(u32, u32)` | Number of cells per block `(cols, rows)`. |
| `nbins` | `u32` | Number of orientation histogram bins (typically 9). |

### Methods

#### `compute(image, device)`

Extracts the HOG feature descriptor from a grayscale image.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `&Image<B>` | Input grayscale image. |
| `device` | `&B::Device` | Compute device. |

**Returns:** `Result<Vec<f32>>` — Flat vector of HOG features.

## Algorithm

1. Compute horizontal and vertical gradients using `[-1, 0, 1]` kernels.
2. Calculate gradient magnitude and orientation at each pixel.
3. Accumulate oriented gradients into cells (histograms).
4. Normalize overlapping blocks using L2-norm.
5. Concatenate all block descriptors into the final feature vector.

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();

let image = Image::<Backend>::open("pedestrian.png", &device)?
    .to_grayscale()?;

let hog = HogDescriptor::new(
    (8, 8),   // cell size
    (2, 2),   // block size
    9,        // bins
);

let features = hog.compute(&image, &device)?;
println!("HOG feature vector length: {}", features.len());
```

## Notes

- Input should be grayscale. Convert with `to_grayscale()` first.
- The classic pedestrian detector uses `cell_size=(8,8)`, `block_size=(2,2)`, `nbins=9`.
- Feature vector length: `(width/8 - 1) * (height/8 - 1) * 2 * 2 * 9`.
