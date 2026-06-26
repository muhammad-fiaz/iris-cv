---
title: "Photo Processing Reference"
description: "API reference for Iris photo processing — NLM denoising and Mertens exposure fusion."
keywords: ["photo", "denoising", "NLM", "non-local means", "exposure fusion", "Mertens", "HDR"]
canonical: "https://muhammad-fiaz.github.io/iris-cv/api/photo"
---

# Photo Processing Reference

High-level photo processing utilities: denoising and exposure fusion.

::: note
This module is under active development. API signatures may change between versions.
:::

## Photo

```rust
pub struct Photo;

impl Photo {
    pub fn fast_nl_means_denoising<B: Backend>(
        image: &Image<B>,
        h: f32,
        patch_radius: usize,
        search_radius: usize,
    ) -> Result<Image<B>>;
}
```

### `fast_nl_means_denoising()`

Applies Non-Local Means (NLM) denoising to a single image using patch-based similarity.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `&Image<B>` | Input image (grayscale or color), values in [0, 1]. |
| `h` | `f32` | Filter strength. Higher values remove more noise but blur detail. Typical: 3.0–10.0. |
| `patch_radius` | `usize` | Half-size of the comparison patch (default: 3 → 7×7 patch). |
| `search_radius` | `usize` | Half-size of the search window (default: 5 → 11×11 window). |

**Returns:** `Result<Image<B>>` — Denoised image.

## MergeMertens

Exposure fusion using the Mertens algorithm. Combines multiple differently-exposed images into a single HDR-like result without requiring exposure metadata.

```rust
pub struct MergeMertens {
    pub contrast_weight: f32,
    pub saturation_weight: f32,
    pub exposure_weight: f32,
}

impl MergeMertens {
    pub fn new() -> Self;
    pub fn with_contrast_weight(self, w: f32) -> Self;
    pub fn with_saturation_weight(self, w: f32) -> Self;
    pub fn with_exposure_weight(self, w: f32) -> Self;
    pub fn process<B: Backend>(&self, images: &[Image<B>]) -> Result<Image<B>>;
}
```

### Constructor

#### `new()`

Creates a new merger with default weights (1.0, 1.0, 1.0). Use builder methods to customize:

```rust
let merger = MergeMertens::new()
    .with_contrast_weight(1.0)
    .with_saturation_weight(1.0)
    .with_exposure_weight(1.0);
```

### Methods

#### `process(images)`

Fuses multiple exposures into a single output. All images must have the same dimensions and 3 channels.

| Parameter | Type | Description |
|-----------|------|-------------|
| `images` | `&[Image<B>]` | Slice of images with different exposures. |

**Returns:** `Result<Image<B>>` — Fused image.

## Example: Denoising

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();

let noisy = Image::<Backend>::open("noisy.png", &device)?;
let clean = Photo::fast_nl_means_denoising(&noisy, 5.0, 3, 5)?;
clean.save("denoised.png")?;
```

## Example: Exposure Fusion

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();

let under = Image::<Backend>::open("underexposed.png", &device)?;
let normal = Image::<Backend>::open("normal.png", &device)?;
let over = Image::<Backend>::open("overexposed.png", &device)?;

let merger = MergeMertens::new();
let fused = merger.process(&[under, normal, over])?;
fused.save("hdr_result.png")?;
```

## Mertens Algorithm

The Mertens fusion computes quality measures (well-exposedness, contrast, saturation) for each input, generates per-pixel weights, and blends the images using weighted averaging. This avoids the need for explicit HDR response curve calibration.
