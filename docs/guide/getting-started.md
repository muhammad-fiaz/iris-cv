---
title: "Getting Started with Iris"
description: "Quick start guide for Iris — install, load an image, apply Gaussian blur, detect Canny edges, and save results in minutes."
keywords: ["getting started", "quick start", "install iris", "first project", "tutorial"]
---

> [!NOTE]
> This project is still in active development. APIs and features may change before the first stable release.

# Getting Started

To get started with **Iris**, install it in your project:

```bash
cargo add iris-cv
```

Alternatively, to use the latest development version from GitHub, run:

```bash
cargo add iris-cv --git https://github.com/muhammad-fiaz/iris-cv
```

## Basic Example

Here is a simple example showing how to load an image, apply a Gaussian blur filter, detect edges using Canny, and save the result:

```rust
use iris::prelude::*;
use burn::backend::wgpu::{Wgpu, WgpuDevice};

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    // Load an image from file
    let img: Image<Backend> = Image::open("input.jpg", &device)?;

    // Smooth the image using Gaussian blur
    let blurred = img.gaussian_blur(5, 1.5)?;

    // Extract edge outlines using Canny detector
    let edges = blurred.grayscale()?.canny(0.1, 0.3)?;

    // Save the output image
    edges.save("edges_output.png")?;
    println!("Successfully processed and saved output!");

    Ok(())
}
```

## What's Next?

- Learn about [Image Representation](/guide/image) to understand how Iris handles images.
- Explore [Filters](/guide/filters) for smoothing and noise reduction.
- See [Edge Detection](/guide/edges) for gradient and boundary detection.
- Browse the [API Reference](/api/) for complete function signatures.