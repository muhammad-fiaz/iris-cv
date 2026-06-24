---
title: "Getting Started with Iris"
description: "Quick start guide for Iris — install, load an image, apply Gaussian blur, detect Canny edges, and save results in minutes."
keywords: ["getting started", "quick start", "install iris", "first project", "tutorial"]
---

# Getting Started

To get started with **Iris**, install it in your project:

```bash
cargo add iris
```

Alternatively, to use the latest development version from GitHub, run:

```bash
cargo add iris --git https://github.com/muhammad-fiaz/iris
```



## Basic Example

Here is a simple example showing how to load an image, apply a Gaussian blur filter, detect edges using Canny, and save the result:

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

fn main() -> Result<()> {
    // 1. Define Burn device
    let device = Default::default();

    // 2. Load an image from file
    let img: Image<Wgpu> = Image::open("input.jpg", &device)?;

    // 3. Smooth the image using Gaussian blur
    let blurred = img.gaussian_blur(5, 1.5)?;

    // 4. Extract edge outlines using Canny detector
    let edges = blurred.grayscale()?.canny(50.0, 150.0)?;

    // 5. Save the output image
    edges.save("edges_output.png")?;
    println!("Successfully processed and saved output!");

    Ok(())
}
```
