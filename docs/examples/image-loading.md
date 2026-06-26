---
title: "Image Loading Example"
description: "Load, resize, annotate, and save images with Iris."
keywords: ["image loading", "image save", "resize", "annotate"]
---

# Image Loading

Loads a real image, resizes it, adds annotations, and saves the result.

```bash
cargo run --example image_loading --features wgpu
```

## Source

```rust
// Demonstrates loading a real image, resizing, grayscale conversion,
// drawing annotations, and saving the result.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Load a real image from assets
    let image: Image<Backend> = Image::open("assets/images/gradient.png", &device)?;
    println!(
        "Original: {}x{} ({} channels)",
        image.width(),
        image.height(),
        image.channels()
    );

    // 2. Process using method chaining
    let processed = image
        .resize(400, 300)?
        .grayscale()?
        .to_rgb()?
        .draw_rectangle(
            Rect::new(50, 50, 200, 150),
            Scalar::new(1.0, 0.0, 0.0, 0.0),
            3,
        )?
        .draw_text(
            "Iris CV",
            Point::new(60, 80),
            2,
            Scalar::new(0.0, 1.0, 0.0, 0.0),
        )?;

    println!(
        "Processed: {}x{} ({} channels)",
        processed.width(),
        processed.height(),
        processed.channels()
    );

    // 3. Save the result
    processed.save("output_image_loading.png")?;
    println!("Saved processed image to 'output_image_loading.png'");

    Ok(())
}
```
