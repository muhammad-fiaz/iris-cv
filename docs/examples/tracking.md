---
title: "Tracking Example"
description: "Background subtraction and CSRT object tracking with Iris."
keywords: ["tracking", "background subtraction", "CSRT"]
---

# Tracking

Demonstrates background subtraction and CSRT object tracking across frames.

```bash
cargo run --example tracking --features wgpu
```

## Source

```rust
// Demonstrates background subtraction and object tracking (CSRT tracker).
// Loads real images as consecutive frames.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load two real images as frames
    let img1: Image<Backend> = Image::open("assets/images/checkerboard.png", &device)?;
    let img2: Image<Backend> = Image::open("assets/images/test_pattern.png", &device)?;

    // 1. Background subtraction
    println!("Applying BackgroundSubtractor...");
    let mut bs = BackgroundSubtractor::new(0.1, 0.05);
    let _mask1 = bs.apply(&img1)?;
    let mask2 = bs.apply(&img2)?;

    println!("Foreground mask shape: {:?}", mask2.shape());
    mask2.save("output_tracking_mask.png")?;

    // 2. Object Tracking
    println!("Initializing CSRT tracker...");
    let mut tracker = Tracker::new(TrackerType::CSRT);
    let init_bbox = Rect::new(50, 50, 100, 100);
    tracker.init(&img1, init_bbox)?;

    println!("Updating tracker with next frame...");
    let updated_bbox = tracker.update(&img2)?;
    println!("Updated bounding box: {:?}", updated_bbox);

    println!("Tracking example completed.");
    Ok(())
}
```
