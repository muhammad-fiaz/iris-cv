---
title: "Stereo Vision Reference"
description: "API reference for Iris stereo vision module — block matching for disparity computation."
keywords: ["stereo vision", "block matching", "disparity", "depth estimation"]
canonical: "https://muhammad-fiaz.github.io/iris-cv/api/stereo"
---

# Stereo Vision Reference

Provides stereo block matching for disparity map computation from rectified stereo image pairs.

::: note
This module is under active development. API signatures may change between versions.
:::

## StereoBlockMatcher

```rust
pub struct StereoBlockMatcher {
    block_size: i32,
    num_disparities: i32,
    min_disparity: i32,
}

impl StereoBlockMatcher {
    pub fn new(block_size: i32, num_disparities: i32) -> Result<Self>;
    pub fn with_min_disparity(self, min_disparity: i32) -> Self;
    pub fn compute<B: Backend>(
        &self,
        left: &Image<B>,
        right: &Image<B>,
    ) -> Result<Tensor<B, 2>>;
}
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `block_size` | `i32` | Size of the matching block (odd number, e.g., 3, 5, 7). |
| `num_disparities` | `i32` | Maximum disparity range (must be divisible by 16). |
| `min_disparity` | `i32` | Minimum disparity to search (default 0). |

### Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();

let left = Image::<Backend>::open("left.png", &device)?;
let right = Image::<Backend>::open("right.png", &device)?;

let matcher = StereoBlockMatcher::new(5, 64)?;
let disparity = matcher.compute(&left, &right)?;
println!("Disparity map shape: {:?}", disparity.dims());
```
