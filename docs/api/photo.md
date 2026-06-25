---
title: "Photo Processing Reference"
description: "API reference for Iris photo processing — NLM denoising and Mertens exposure fusion."
keywords: ["photo", "denoising", "NLM", "non-local means", "exposure fusion", "Mertens", "HDR"]
canonical: "https://muhammad-fiaz.github.io/iris/api/photo"
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
        filter_strength: f32,
        template_window_size: u32,
        search_window_size: u32,
        device: &B::Device,
    ) -> Result<Image<B>>;
}
```

### `fast_nl_means_denoising()`

Applies Non-Local Means (NLM) denoising to a single image.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `&Image<B>` | Input image (grayscale or color). |
| `filter_strength` | `f32` | Strength of the filter. Higher values remove more noise but blur detail. Typical: 3.0–10.0. |
| `template_window_size` | `u32` | Size of the patch used for comparison (odd). Typical: 7. |
| `search_window_size` | `u32` | Size of the search area (odd). Typical: 21. |
| `device` | `&B::Device` | Compute device. |

**Returns:** `Result<Image<B>>` — Denoised image.

## MergeMertens

Exposure fusion using the Mertens algorithm. Combines multiple differently-exposed images into a single HDR-like result without requiring exposure metadata.

```rust
pub struct MergeMertens {
    contrast_weight: f32,
    saturation_weight: f32,
    exposure_weight: f32,
}

impl MergeMertens {
    pub fn new(
        contrast_weight: f32,
        saturation_weight: f32,
        exposure_weight: f32,
    ) -> Self;

    pub fn process<B: Backend>(
        &self,
        images: &[Image<B>],
        device: &B::Device,
    ) -> Result<Image<B>>;
}
```

### Constructor

#### `new(contrast_weight, saturation_weight, exposure_weight)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `contrast_weight` | `f32` | Weight for well-exposedness measure. |
| `saturation_weight` | `f32` | Weight for color saturation. |
| `exposure_weight` | `f32` | Weight for contrast measure. |

Use equal weights `(1.0, 1.0, 1.0)` for balanced fusion.

### Methods

#### `process(images, device)`

Fuses multiple exposures into a single output.

| Parameter | Type | Description |
|-----------|------|-------------|
| `images` | `&[Image<B>]` | Slice of images with different exposures. |
| `device` | `&B::Device` | Compute device. |

**Returns:** `Result<Image<B>>` — Fused image.

## Example: Denoising

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();

let noisy = Image::<Backend>::open("noisy.png", &device)?;
let clean = Photo::fast_nl_means_denoising(&noisy, 5.0, 7, 21, &device)?;
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

let merger = MergeMertens::new(1.0, 1.0, 1.0);
let fused = merger.process(&[under, normal, over], &device)?;
fused.save("hdr_result.png")?;
```

## Mertens Algorithm

The Mertens fusion computes quality measures (well-exposedness, contrast, saturation) for each input, generates per-pixel weights, and blends the images using weighted averaging. This avoids the need for explicit HDR response curve calibration.
