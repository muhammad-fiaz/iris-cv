---
title: "Segmentation Example"
description: "Semantic segmentation and connected components with Iris."
keywords: ["segmentation", "connected components"]
---

# Segmentation

Demonstrates semantic segmentation and connected components labeling.

```bash
cargo run --example segmentation --features wgpu
```

## Source

```rust
// Demonstrates semantic segmentation and connected components labeling.
// Loads a real image for segmentation and uses a synthetic binary image for CC.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Load a real image for segmentation
    let img: Image<Backend> = Image::open("assets/images/gradient.png", &device)?;

    // 2. Semantic segmentation
    println!("Running semantic segmentation...");
    let segmenter = Segmenter::<Backend>::default();
    let mask = segmenter.segment(&img)?;
    println!("Segmentation mask shape: {:?}", mask.mask.dims());

    // 3. Connected components labeling on a synthetic binary image
    println!("Running connected components labeling...");
    let mut binary_data = vec![0.0f32; 50 * 50];
    for y in 20..25 {
        for x in 20..25 {
            binary_data[y * 50 + x] = 1.0;
        }
    }
    let binary_img = Image::new(Tensor::<Backend, 3>::from_data(
        TensorData::new(binary_data, [1, 50, 50]),
        &device,
    ));
    let (_labels, stats) = binary_img.connected_components_with_stats()?;

    println!("Found {} connected component(s):", stats.len());
    for stat in &stats {
        println!(
            "  - Label: {}, bbox: {:?}, area: {}, centroid: {:?}",
            stat.label, stat.bbox, stat.area, stat.centroid
        );
    }

    img.save("output_segmentation.png")?;
    println!("Saved input image to 'output_segmentation.png'");

    Ok(())
}
```
