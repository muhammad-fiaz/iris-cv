---
title: "Optical Flow Example"
description: "Dense (Farneback) and sparse (Lucas-Kanade) optical flow with Iris."
keywords: ["optical flow", "Farneback", "Lucas-Kanade", "motion"]
---

# Optical Flow

Demonstrates dense and sparse optical flow computation on two real images.

```bash
cargo run --example optical_flow --features wgpu
```

## Source

```rust
// Demonstrates dense (Farneback) and sparse (Lucas-Kanade) optical flow.
// Loads two real images and computes motion between them.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load two real images as consecutive frames
    let img1: Image<Backend> = Image::open("assets/images/checkerboard.png", &device)?;
    let img2: Image<Backend> = Image::open("assets/images/test_pattern.png", &device)?;

    // Dense Optical Flow (Farneback)
    println!("Calculating dense optical flow (Farneback)...");
    let flow = OpticalFlow::calc_dense_farneback(&img1, &img2)?;
    println!("Dense flow tensor shape: {:?}", flow.dims());

    // Sparse Optical Flow (Lucas-Kanade)
    println!("Calculating sparse optical flow (Lucas-Kanade)...");
    let prev_pts = vec![
        Point::new(100.0, 100.0),
        Point::new(200.0, 200.0),
        Point::new(300.0, 300.0),
    ];
    let (next_pts, status) = OpticalFlow::calc_sparse_pyr_lk(&img1, &img2, &prev_pts)?;
    for i in 0..prev_pts.len() {
        if status[i] == 1 {
            println!(
                "  Point tracked from {:?} to {:?}",
                prev_pts[i], next_pts[i]
            );
        } else {
            println!("  Point {:?} lost tracking", prev_pts[i]);
        }
    }

    println!("Optical flow example completed.");
    Ok(())
}
```
