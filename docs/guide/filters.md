---
title: "Image Filters & Blur"
description: "Apply box blur, Gaussian blur, median filter, bilateral filter, separable 2D filters, distance transform, and more in Rust with Iris."
keywords: ["image filters", "blur", "gaussian blur", "median filter", "bilateral filter", "box blur", "smoothing"]
---

# Image Filters & Blur

Image filtering is essential for noise reduction, smoothing, and feature enhancement. Iris implements standard spatial filters using parallelized row-by-row CPU iterations (powered by `rayon` under the `parallel` feature flag) or accelerated GPU tensor calculations.

## Available Blur Filters

### Box Blur

Smooths an image using a normalized box filter of the given kernel size.

```rust
// Applies a 5x5 box filter
let blurred = image.box_blur(5)?;
```

### Gaussian Blur

Smooths an image using a Gaussian kernel. You must specify the kernel size (must be odd) and standard deviation `sigma`.

```rust
// Applies a 5x5 Gaussian blur with sigma = 1.5
let gaussian = image.gaussian_blur(5, 1.5)?;
```

### Median Blur

Reduces salt-and-pepper noise by taking the median value in each local neighborhood.

```rust
// Applies a 5x5 median filter
let median = image.median_blur(5)?;
```

### Bilateral Filter

Smooths the image while preserving sharp edges. It uses a range sigma for color similarity and space sigma for coordinate closeness.

```rust
// Applies a bilateral filter with d=5, sigma_color=0.1, sigma_space=10.0
let filtered = image.bilateral_filter(5, 0.1, 10.0)?;
```

### Separable 2D Filter

Applies two 1D kernels sequentially along the X and Y dimensions to achieve efficient custom 2D filtering.

```rust
let kernel_x = vec![0.25, 0.5, 0.25];
let kernel_y = vec![0.25, 0.5, 0.25];
let filtered = image.sep_filter_2d(&kernel_x, &kernel_y)?;
```

## Utility Filters

### Filter2D

Applies a general 2D convolution kernel to the image.

```rust
let kernel: &[&[f32]] = &[
    &[0.0, -1.0, 0.0],
    &[-1.0, 5.0, -1.0],
    &[0.0, -1.0, 0.0],
];
let sharpened = image.filter2d(kernel, None, 0.0)?;
```

### Add Weighted

Blends two images: `result = alpha * src1 + beta * src2 + gamma`.

```rust
let blended = img1.add_weighted(&img2, 0.7, 0.3, 0.0)?;
```

### Convert Scale Abs

Converts the image with per-pixel scale and shift, then takes absolute values.

```rust
let abs_img = image.convert_scale_abs(1.0, 0.0)?;
```

### Distance Transform

Computes the distance transform of a binary image, where each pixel value is its distance to the nearest zero pixel.

```rust
let dist = image.distance_transform()?;
```
