---
title: "Edge Detection & Gradients"
description: "Detect edges and compute gradients using Canny, Sobel, Scharr, Laplacian, Hough lines, and Hough circles in Rust with Iris."
keywords: ["edge detection", "canny", "sobel", "scharr", "laplacian", "gradients", "hough lines", "hough circles"]
---

# Edge Detection & Gradients

Gradients and edge maps highlight structural changes and borders within images. Iris provides standard first and second-order image derivative approximations, plus geometric shape detectors.

## Gradient Operators

### Sobel

Calculates first-order derivatives along the X and Y axes. Returns a single gradient magnitude image.

```rust
// Compute Sobel gradient magnitude
let edges = image.sobel()?;
```

### Scharr

Provides more accurate derivative estimations compared to the Sobel operator.

```rust
// Compute Scharr gradient magnitude
let scharr_img = image.scharr()?;
```

### Laplacian

Calculates the second-order derivative (sum of second spatial derivatives), emphasizing high-frequency regions.

```rust
// Compute Laplacian
let laplacian = image.laplacian()?;
```

## Canny Edge Detection

Canny edge detection is a multi-stage process that reduces noise, computes gradients, performs non-maximum suppression, and uses hysteresis thresholding.

```rust
// Run Canny edge detector with low/high thresholds
let edges = image.canny(0.1, 0.3)?;
```

## Hough Transforms

### Probabilistic Hough Line Transform

Detects line segments in a binary edge image. Returns a vector of `LineSegment` as `((x1,y1), (x2,y2))`.

```rust
let lines = edges.hough_lines_p(
    1.0,                            // rho: distance resolution
    std::f32::consts::PI / 180.0,   // theta: angle resolution
    50,                             // threshold: minimum votes
    50,                             // min_line_length
    10,                             // max_line_gap
)?;
for ((x1, y1), (x2, y2)) in lines {
    println!("Line from ({}, {}) to ({}, {})", x1, y1, x2, y2);
}
```

### Hough Circle Transform

Detects circles in a grayscale image. Returns circles as `(center_x, center_y, radius)`.

```rust
let circles = image.hough_circles(
    1.0,    // dp: accumulator resolution ratio
    20.0,   // min_dist: minimum distance between circles
    50.0,   // param1: upper Canny threshold
    30.0,   // param2: accumulator threshold
    10,     // min_radius
    100,    // max_radius
)?;
for (cx, cy, r) in circles {
    println!("Circle at ({}, {}) with radius {}", cx, cy, r);
}
```
