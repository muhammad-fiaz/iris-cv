---
title: "Color Module Reference"
description: "API reference for Iris color module — RGB/HSV/HLS/XYZ/Lab/YUV/YCrCb/CMYK/HSL color space conversions and channel splitting/merging."
keywords: ["color", "RGB", "HSV", "HLS", "XYZ", "Lab", "YUV", "YCrCb", "CMYK", "HSL", "color conversion", "split channels", "merge channels", "rgb_to_cmyk", "rgb_to_hsl"]
---

# Color Module Reference

Provides color space conversion utilities for images. All conversions operate on 3-channel images with values in `[0.0, 1.0]`.

## Color Space Conversions

```rust
impl<B: Backend> Image<B> {
    pub fn rgb_to_hsv(&self) -> Result<Self>;
    pub fn hsv_to_rgb(&self) -> Result<Self>;
    pub fn rgb_to_hls(&self) -> Result<Self>;
    pub fn hls_to_rgb(&self) -> Result<Self>;
    pub fn rgb_to_xyz(&self) -> Result<Self>;
    pub fn xyz_to_rgb(&self) -> Result<Self>;
    pub fn rgb_to_lab(&self) -> Result<Self>;
    pub fn lab_to_rgb(&self) -> Result<Self>;
    pub fn rgb_to_yuv(&self) -> Result<Self>;
    pub fn yuv_to_rgb(&self) -> Result<Self>;
    pub fn rgb_to_ycrcb(&self) -> Result<Self>;
    pub fn ycrcb_to_rgb(&self) -> Result<Self>;
}
```

## Channel Operations

```rust
impl<B: Backend> Image<B> {
    pub fn split_channels(&self) -> Result<Vec<Self>>;
    pub fn merge_channels(channels: &[Image<B>]) -> Result<Self>;
}
```

## Color Space Details

| Space | Channels | Range | Description |
|---|---|---|---|
| RGB | 3 | [0, 1] | Standard Red-Green-Blue. |
| HSV | 3 | H:[0,1], S:[0,1], V:[0,1] | Hue (normalized from 360), Saturation, Value. |
| HLS | 3 | H:[0,1], L:[0,1], S:[0,1] | Hue, Lightness, Saturation. |
| XYZ | 3 | [0, 1] | CIE XYZ (D65 illuminant). |
| Lab | 3 | [0, 1] | CIE L\*a\*b\* (D65 white point). L normalized to [0,100]. |
| YUV | 3 | [0, 1] | BT.601 luma + chroma. |
| YCrCb | 3 | [0, 1] | BT.601 luma + red/blue chroma. |

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();
let img = Image::<Backend>::open("photo.jpg", &device)?;

// Convert to HSV, manipulate, convert back
let hsv = img.rgb_to_hsv()?;
let result = hsv.hsv_to_rgb()?;

// Split into channels and merge back
let channels = img.split_channels()?;
let merged = Image::merge_channels(&channels)?;
```

## CMYK Conversions

```rust
impl<B: Backend> Image<B> {
    pub fn rgb_to_cmyk(&self) -> Result<Self>;
    pub fn cmyk_to_rgb(&self) -> Result<Self>;
}
```

Converts between RGB and CMYK (Cyan, Magenta, Yellow, Key/Black) color spaces. The CMYK image has 4 channels, each in `[0.0, 1.0]`.

## HSL Conversions

```rust
impl<B: Backend> Image<B> {
    pub fn rgb_to_hsl(&self) -> Result<Self>;
    pub fn hsl_to_rgb(&self) -> Result<Self>;
}
```

Converts between RGB and HSL (Hue, Saturation, Lightness) color spaces. The HSL image has 3 channels: H in `[0.0, 1.0]`, S in `[0.0, 1.0]`, L in `[0.0, 1.0]`.
