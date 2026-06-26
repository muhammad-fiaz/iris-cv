---
title: "Image Stitching Reference"
description: "API reference for Iris image stitching — panorama creation from multiple images."
keywords: ["stitching", "panorama", "image stitching", "stitch"]
canonical: "https://muhammad-fiaz.github.io/iris-cv/api/stitching"
---

# Image Stitching Reference

Provides image stitching for creating panoramas from multiple overlapping images.

::: note
This module is under active development. API signatures may change between versions.
:::

## Stitcher

```rust
pub struct Stitcher;

impl Stitcher {
    pub fn stitch<B: Backend>(&self, images: &[Image<B>]) -> Result<Image<B>>;
}
```

### Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();

let img1 = Image::<Backend>::open("left.jpg", &device)?;
let img2 = Image::<Backend>::open("right.jpg", &device)?;

let stitcher = Stitcher;
let panorama = stitcher.stitch(&[img1, img2])?;
panorama.save("panorama.jpg")?;
```
