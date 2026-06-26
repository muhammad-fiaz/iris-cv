---
title: "Camera Calibration Example"
description: "Camera calibration, 3D-to-2D projection, and homography with Iris."
keywords: ["camera calibration", "homography", "projection"]
---

# Camera Calibration

Demonstrates camera calibration, 3D-to-2D point projection, and homography estimation using synthetic point correspondences.

```bash
cargo run --example camera_calibration --features wgpu
```

## Source

```rust
{{#include ../../../examples/camera_calibration.rs}}
```
