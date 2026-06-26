---
title: "Filters Example"
description: "Box, Gaussian, median, bilateral, and separable filters with Iris."
keywords: ["filters", "blur", "gaussian", "median", "bilateral"]
---

# Filters

Demonstrates various image filtering operations on a real image.

```bash
cargo run --example filters --features wgpu
```

## Source

```rust
// Demonstrates various image filtering operations:
// box blur, Gaussian blur, median blur, bilateral filter, and separable filter.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load a real image
    let image: Image<Backend> = Image::open("assets/images/color_bars.png", &device)?;
    println!(
        "Loaded image: {}x{} ({} channels)",
        image.width(),
        image.height(),
        image.channels()
    );

    // Box Blur
    println!("Applying Box Blur...");
    let box_blurred = image.clone().box_blur(5)?;
    box_blurred.save("output_filters_box_blur.png")?;

    // Gaussian Blur
    println!("Applying Gaussian Blur...");
    let gaussian_blurred = image.clone().gaussian_blur(5, 1.0)?;
    gaussian_blurred.save("output_filters_gaussian_blur.png")?;

    // Median Blur
    println!("Applying Median Blur...");
    let median_blurred = image.clone().median_blur(5)?;
    median_blurred.save("output_filters_median_blur.png")?;

    // Bilateral Filter
    println!("Applying Bilateral Filter...");
    let bilateral = image.clone().bilateral_filter(5, 0.1, 10.0)?;
    bilateral.save("output_filters_bilateral.png")?;

    // Separable Filter
    println!("Applying Separable Filter...");
    let sep = image
        .clone()
        .sep_filter_2d(&[0.2, 0.6, 0.2], &[0.2, 0.6, 0.2])?;
    sep.save("output_filters_sep.png")?;

    println!("All filter operations completed. Outputs saved to output_filters_*.png");
    Ok(())
}
```
