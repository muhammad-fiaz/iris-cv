---
title: "Image Stitching Reference"
description: "API reference for Iris image stitching — homography-based panorama stitching with ORB features and RANSAC."
keywords: ["stitching", "panorama", "homography", "RANSAC", "feature matching", "warping"]
canonical: "https://muhammad-fiaz.github.io/iris/api/stitching"
---

# Image Stitching Reference

Homography-based panorama stitching from overlapping image sequences.

::: note
This module is under active development. API signatures may change between versions.
:::

## Stitcher

```rust
pub struct Stitcher {
    match_ratio: f32,
    ransac_threshold: f32,
    max_iterations: u32,
}

impl Stitcher {
    pub fn new() -> Self;
    pub fn stitch<B: Backend>(
        &self,
        images: &[Image<B>],
        device: &B::Device,
    ) -> Result<Image<B>>;
    pub fn set_match_ratio(&mut self, ratio: f32);
    pub fn set_ransac_threshold(&mut self, threshold: f32);
    pub fn set_max_iterations(&mut self, iterations: u32);
}
```

### Constructor

#### `new()`

Creates a `Stitcher` with default parameters:
- `match_ratio`: 0.75 (Lowe's ratio test threshold)
- `ransac_threshold`: 5.0 (pixel reprojection tolerance)
- `max_iterations`: 1000 (RANSAC iterations)

### Configuration Methods

| Method | Description |
|--------|-------------|
| `set_match_ratio(ratio)` | Lowe's ratio test threshold for feature matching (0.0–1.0). |
| `set_ransac_threshold(threshold)` | Maximum reprojection error in pixels for RANSAC inlier classification. |
| `set_max_iterations(iterations)` | Maximum RANSAC iterations. |

### Methods

#### `stitch(images, device)`

Stitches a sequence of overlapping images into a panorama.

| Parameter | Type | Description |
|-----------|------|-------------|
| `images` | `&[Image<B>]` | Ordered slice of images to stitch (left to right). |
| `device` | `&B::Device` | Compute device. |

**Returns:** `Result<Image<B>>` — Stitched panorama image.

## Pipeline

The stitching pipeline executes the following stages:

1. **Feature Detection** — ORB features (keypoints + descriptors) are extracted from each image.
2. **Feature Matching** — Brute-force Hamming matcher with Lowe's ratio test filters reliable correspondences.
3. **Homography Estimation** — Direct Linear Transform (DLT) computes candidate homographies from matched point pairs.
4. **RANSAC Outlier Rejection** — Iterative RANSAC refines the homography by classifying inliers based on reprojection error.
5. **Image Warping** — Each image is warped to the reference frame using the computed homography.
6. **Blending** — Warped images are composited with overlap blending to produce the final panorama.

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();

let img1 = Image::<Backend>::open("left.png", &device)?;
let img2 = Image::<Backend>::open("center.png", &device)?;
let img3 = Image::<Backend>::open("right.png", &device)?;

let mut stitcher = Stitcher::new();
stitcher.set_ransac_threshold(3.0);

let panorama = stitcher.stitch(&[img1, img2, img3], &device)?;
panorama.save("panorama.png")?;
```

## Notes

- Images should have sufficient overlap (30–50%) for reliable feature matching.
- ORB features work best on textured scenes. Featureless or repetitive textures may fail.
- For best results, images should be roughly aligned horizontally and taken from the same camera position.
