---
title: "Stereo Vision Module Reference"
description: "API reference for Iris stereo vision — block matching disparity computation."
keywords: ["stereo", "disparity", "block matching", "SAD", "depth", "stereo vision"]
canonical: "https://muhammad-fiaz.github.io/iris/api/stereo"
---

# Stereo Vision Module Reference

Computes disparity maps from stereo image pairs using block matching.

::: note
This module is under active development. API signatures may change between versions.
:::

## StereoBlockMatcher

Block-matching stereo correspondence engine using Sum of Absolute Differences (SAD).

```rust
pub struct StereoBlockMatcher {
    num_disparities: u32,
    block_size: u32,
}

impl StereoBlockMatcher {
    pub fn new(num_disparities: u32, block_size: u32) -> Result<Self>;
    pub fn compute<B: Backend>(
        &self,
        left: &Image<B>,
        right: &Image<B>,
        device: &B::Device,
    ) -> Result<Image<B>>;
}
```

### Constructor

#### `new(num_disparities, block_size)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_disparities` | `u32` | Maximum disparity search range. Must be a multiple of 16. |
| `block_size` | `u32` | Size of the matching block (kernel). Must be odd. |

### Methods

#### `compute(left, right, device)`

Computes a disparity map from a rectified stereo pair.

| Parameter | Type | Description |
|-----------|------|-------------|
| `left` | `&Image<B>` | Left rectified grayscale image. |
| `right` | `&Image<B>` | Right rectified grayscale image. |
| `device` | `&B::Device` | Compute device for GPU acceleration. |

**Returns:** `Result<Image<B>>` — Single-channel disparity map where intensity encodes pixel shift.

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();

let left = Image::<Backend>::open("left.png", &device)?.to_grayscale()?;
let right = Image::<Backend>::open("right.png", &device)?.to_grayscale()?;

let matcher = StereoBlockMatcher::new(64, 11)?;
let disparity = matcher.compute(&left, &right, &device)?;
disparity.save("disparity.png")?;
```

## Notes

- Both input images must be rectified (epipolar aligned).
- `num_disparities` must be divisible by 16 for efficient GPU kernel dispatch.
- Larger `block_size` values improve noise tolerance but reduce detail.
