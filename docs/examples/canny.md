---
title: "Canny Edge Detection Example"
description: "Canny edge detection on real images with Iris."
keywords: ["canny", "edge detection"]
---

# Canny Edge Detection

Loads the test pattern image and applies Canny edge detection with two threshold levels.

```bash
cargo run --example canny --features wgpu
```

## Source

```rust
// Demonstrates Canny edge detection on a real image.
// Loads the test pattern and applies Canny with two threshold levels.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load the test pattern image
    let image: Image<Backend> = Image::open("assets/images/test_pattern.png", &device)?;
    println!(
        "Loaded image: {}x{} ({} channels)",
        image.width(),
        image.height(),
        image.channels()
    );

    // Perform Canny edge detection
    let edges = image.canny(0.1, 0.4)?;
    println!("Edges shape: {:?}", edges.shape());

    // Save the result
    edges.save("output_canny.png")?;
    println!("Saved Canny edges to 'output_canny.png'");

    Ok(())
}
```
