---
title: "ArUco Pose Example"
description: "ArUco marker detection and pose estimation with Iris."
keywords: ["ArUco", "marker", "pose estimation"]
---

# ArUco Pose

Demonstrates ArUco marker detection and single-marker pose estimation.

```bash
cargo run --example aruco_pose --features wgpu
```

## Source

```rust
// Demonstrates ArUco marker detection and single-marker pose estimation.
// Generates a synthetic frame (no real ArUco marker), detects markers,
// and prints estimated rotation/translation vectors.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load a real image as the background frame
    let frame: Image<Backend> = Image::open("assets/images/checkerboard.png", &device)?;

    // Detect ArUco markers
    let detector = ArucoDetector::new(ArucoDict::Dict6X6_250);
    let markers = detector.detect_markers(&frame)?;

    println!("Detected {} ArUco marker(s):", markers.len());
    for m in &markers {
        println!("  - Marker ID: {}, corners: {:?}", m.id, m.corners);
    }

    // Estimate single marker poses (if markers found)
    if !markers.is_empty() {
        let camera_matrix = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]];
        let dist_coeffs = vec![0.0; 5];

        let (rvecs, tvecs) =
            detector.estimate_pose_single_markers(&markers, 0.05, &camera_matrix, &dist_coeffs)?;
        for i in 0..markers.len() {
            println!(
                "  Marker {}: tvec={:?}, rvec={:?}",
                markers[i].id, tvecs[i], rvecs[i]
            );
        }
    }

    frame.save("output_aruco_pose.png")?;
    println!("Saved frame to 'output_aruco_pose.png'");

    Ok(())
}
```
