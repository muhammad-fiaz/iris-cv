---
title: "Motion Tracking & Optical Flow"
description: "Track objects with KCF, CSRT, and MOSSE trackers. Compute dense Farneback and sparse Lucas-Kanade optical flow."
keywords: ["tracking", "optical flow", "background subtraction", "KCF", "CSRT", "Farneback", "Lucas-Kanade"]
---

# Motion Tracking & Optical Flow

Iris includes algorithms to isolate moving objects, track dynamic structures, and evaluate pixel motion vectors.

## Background Subtractor

A pipeline that models stationary scenes and isolates transient moving foreground elements by computing running averages and absolute differences.

```rust
// Create background subtractor (learning_rate = 0.1, threshold = 0.05)
let mut subtractor = BackgroundSubtractor::new(0.1, 0.05);

// Process frame sequence
let mask = subtractor.apply(&frame)?; // grayscale mask highlighting motion
```

## Object Tracking

Tracks a specified bounding box across subsequent video frames using dynamic updating models. Supported backends include:

- **`TrackerType::KCF`**: Kernelized Correlation Filters.
- **`TrackerType::CSRT`**: Channel and Spatial Reliability Tracking.
- **`TrackerType::MOSSE`**: Minimum Barrier Distance Trackers.

```rust
let mut tracker = Tracker::new(TrackerType::KCF);

// Initialize tracker on first frame
let init_box = Rect::new(50, 50, 100, 100);
tracker.init(&frame1, init_box)?;

// Update on subsequent frames
let updated_box = tracker.update(&frame2)?;
```

## Optical Flow

Optical flow tracks movements of pixels between two images.

### Dense Flow (Farneback)
Calculates flow vectors `(dx, dy)` for all pixels, returning a flow tensor of shape `[2, H, W]`.

```rust
let flow_tensor = OpticalFlow::calc_dense_farneback(&frame1, &frame2)?;
```

### Sparse Flow (Lucas-Kanade)
Tracks specific keypoints across frames using spatial image pyramids.

```rust
let prev_pts = vec![Point::new(10.0, 10.0)];
let (next_pts, status) = OpticalFlow::calc_sparse_pyr_lk(&frame1, &frame2, &prev_pts)?;
```
