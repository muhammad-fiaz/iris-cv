---
title: "Contours Example"
description: "Contour detection, convex hull, and moments with Iris."
keywords: ["contours", "convex hull", "moments"]
---

# Contours

Demonstrates contour detection, convex hull computation, and contour moments.

```bash
cargo run --example contours --features wgpu
```

## Source

```rust
// Demonstrates contour detection, convex hull, and contour moments.
// Loads the checkerboard image which has clear edges for contour finding.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load the checkerboard image (strong contrast edges)
    let image: Image<Backend> = Image::open("assets/images/checkerboard.png", &device)?;

    // Find contours
    let contours = image.find_contours()?;
    println!("Found {} contour(s)", contours.len());

    if let Some(contour) = contours.first() {
        println!("Contour size: {} points", contour.points.len());

        // Compute convex hull
        let hull = contour.convex_hull();
        println!("Convex hull size: {} points", hull.points.len());

        // Compute moments
        let m = contour.moments();
        println!("Contour Area (m00): {}", m.m00);
        if let Some(centroid) = m.centroid() {
            println!("Contour Centroid: ({:.2}, {:.2})", centroid.x, centroid.y);
        }
    }

    image.save("output_contours.png")?;
    println!("Saved input image to 'output_contours.png'");

    Ok(())
}
```
