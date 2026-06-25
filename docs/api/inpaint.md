---
title: "Inpainting Module Reference"
description: "API reference for Iris inpainting — Telea Fast Marching Method for mask-based image inpainting."
keywords: ["inpainting", "inpaint", "telea", "fast marching", "image restoration", "missing pixels"]
canonical: "https://muhammad-fiaz.github.io/iris/api/inpaint"
---

# Inpainting Module Reference

Restores missing or corrupted regions in an image using the Telea Fast Marching Method.

::: note
This module is under active development. API signatures may change between versions.
:::

## Function

```rust
pub fn inpaint<B: Backend>(
    image: &Image<B>,
    mask: &Image<B>,
    radius: f32,
) -> Result<Image<B>>
```

Performs inpainting on `image` using a binary `mask` to identify regions to fill. The `radius` parameter controls the neighborhood size used for pixel propagation.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `&Image<B>` | Source image with missing or corrupted regions. |
| `mask` | `&Image<B>` | Binary mask (same dimensions as `image`) where non-zero pixels mark areas to inpaint. |
| `radius` | `f32` | Radius of the Fast Marching propagation kernel. Larger values consider more surrounding pixels. |

### Returns

`Result<Image<B>>` — The inpainted image with mask regions filled.

## Algorithm

The Telea method treats inpainting as a Fast Marching problem. Starting from the boundary of the masked region, it propagates pixel values inward using a weighted average of known neighbors, controlled by the `radius`.

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();

let image = Image::<Backend>::open("photo.png", &device)?;
let mask = Image::<Backend>::open("mask.png", &device)?;

// Inpaint with a radius of 3 pixels
let restored = inpaint(&image, &mask, 3.0)?;
restored.save("restored.png")?;
```

## Use Cases

- Scratch and dust removal from scanned photos
- Removing unwanted objects from images
- Filling in missing data from occluded regions
- Restoring damaged artwork or historical documents
