---
title: "Threshold Module Reference"
description: "API reference for Iris threshold module — binary, Otsu, triangle, and adaptive thresholding operations."
keywords: ["threshold", "binary threshold", "Otsu", "triangle", "adaptive threshold", "segmentation"]
---

# Threshold Module Reference

Provides thresholding operations for image binarization and segmentation.

## Threshold Types

```rust
pub enum ThresholdType {
    Binary,
    BinaryInv,
    Trunc,
    ToZero,
    ToZeroInv,
}
```

| Type | Description |
|---|---|
| `Binary` | `pixel > thresh ? maxval : 0` |
| `BinaryInv` | `pixel > thresh ? 0 : maxval` |
| `Trunc` | `pixel > thresh ? thresh : pixel` |
| `ToZero` | `pixel > thresh ? pixel : 0` |
| `ToZeroInv` | `pixel > thresh ? 0 : pixel` |

## Adaptive Methods

```rust
pub enum AdaptiveMethod {
    MeanC,
    GaussianC,
}
```

| Method | Description |
|---|---|
| `MeanC` | Mean of the blockSize x blockSize neighborhood. |
| `GaussianC` | Gaussian-weighted sum of the neighborhood. |

## Operations

```rust
impl<B: Backend> Image<B> {
    pub fn threshold(&self, thresh: f32, maxval: f32, thresh_type: ThresholdType) -> Result<Self>;
    pub fn threshold_otsu(&self, maxval: f32) -> Result<Self>;
    pub fn threshold_triangle(&self, maxval: f32) -> Result<Self>;
    pub fn adaptive_threshold(
        &self,
        maxval: f32,
        method: AdaptiveMethod,
        block_size: usize,
        c: f32,
    ) -> Result<Self>;
}
```

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();
let img = Image::<Backend>::open("input.jpg", &device)?;

// Fixed binary threshold
let binary = img.threshold(0.5, 1.0, ThresholdType::Binary)?;

// Automatic Otsu threshold
let otsu = img.threshold_otsu(1.0)?;

// Triangle method
let triangle = img.threshold_triangle(1.0)?;

// Adaptive threshold
let adaptive = img.adaptive_threshold(1.0, AdaptiveMethod::MeanC, 11, 2.0)?;
```
