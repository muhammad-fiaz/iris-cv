---
title: "Histogram Module Reference"
description: "API reference for Iris histogram module — calc_hist, equalize_hist, CLAHE, apply_lut, compare_hist, 2D histogram, adaptive histogram equalization, and color histogram operations."
keywords: ["histogram", "equalization", "CLAHE", "LUT", "compare histogram", "contrast enhancement", "2D histogram", "adaptive equalization", "calc_hist_2d", "equalize_hist_adaptive"]
---

# Histogram Module Reference

Provides histogram computation, equalization, adaptive equalization, lookup tables, and histogram comparison.

## Operations

```rust
impl<B: Backend> Image<B> {
    pub fn calc_hist(&self) -> Result<Vec<Vec<u32>>>;
    pub fn equalize_hist(&self) -> Result<Self>;
    pub fn equalize_hist_color(&self) -> Result<Self>;
    pub fn clahe(&self, clip_limit: f32, grid_size: usize) -> Result<Self>;
    pub fn apply_lut(&self, lut: &[f32; 256]) -> Result<Self>;
}
```

## Histogram Comparison

```rust
impl<B: Backend> Image<B> {
    pub fn compare_hist(hist_a: &[f32], hist_b: &[f32], method: &str) -> Result<f64>;
    pub fn compare_hist_color(
        hist_a: &[Vec<f32>],
        hist_b: &[Vec<f32>],
        method: &str,
    ) -> Result<Vec<f64>>;
}
```

### Comparison Methods

| Method | Description |
|---|---|
| `"correlation"` | Pearson correlation coefficient. Identical histograms yield 1.0. |
| `"chi_square"` | Chi-squared distance. Identical histograms yield 0.0. |
| `"intersection"` | Sum of min(a, b) per bin. Higher = more similar. |
| `"hellinger"` | Hellinger distance. Identical histograms yield 0.0. |

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();
let img = Image::<Backend>::open("input.jpg", &device)?;

// Compute per-channel histograms
let hists = img.calc_hist()?;

// Equalize grayscale histogram for contrast enhancement
let equalized = img.equalize_hist()?;

// Adaptive CLAHE
let clahe_result = img.clahe(2.0, 8)?;

// Color histogram equalization (converts to YCrCb internally)
let color_eq = img.equalize_hist_color()?;

// Invert brightness using LUT
let mut lut = [0.0f32; 256];
for i in 0..256 {
    lut[i] = 1.0 - (i as f32) / 255.0;
}
let inverted = img.apply_lut(&lut)?;

// Compare histograms
let hist_a = img.calc_hist()?;
let hist_b = equalized.calc_hist()?;
let similarity = Image::<Backend>::compare_hist(&hist_a[0], &hist_b[0], "correlation")?;
```

## 2D Histogram

```rust
impl<B: Backend> Image<B> {
    pub fn calc_hist_2d(&self, channel_x: usize, channel_y: usize, bins: usize) -> Result<Tensor<B, 2>>;
}
```

Computes a 2D histogram over two specified channels. Returns a `(bins, bins)` tensor.

| Parameter | Description |
|---|---|
| `channel_x` | Index of the first channel (e.g., 0 for Red). |
| `channel_y` | Index of the second channel (e.g., 1 for Green). |
| `bins` | Number of bins per dimension. |

### Example

```rust,ignore
let hist_2d = img.calc_hist_2d(0, 1, 64)?; // Red vs Green, 64x64 bins
println!("2D histogram shape: {:?}", hist_2d.dims());
```

## Adaptive Histogram Equalization

```rust
impl<B: Backend> Image<B> {
    pub fn equalize_hist_adaptive(&self, grid_size: usize) -> Result<Self>;
}
```

Applies adaptive histogram equalization by dividing the image into a `grid_size x grid_size` grid of tiles and equalizing each independently.

| Parameter | Description |
|---|---|
| `grid_size` | Number of tiles per row/column (e.g., 8 yields an 8×8 grid). |

### Example

```rust,ignore
let adaptive_eq = img.equalize_hist_adaptive(8)?;
```
