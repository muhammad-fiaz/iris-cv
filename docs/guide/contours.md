---
title: "Segmentation & Contours"
description: "Find contours, compute convex hulls, analyze shapes with Hu moments, and label connected components in binary images."
keywords: ["segmentation", "contours", "convex hull", "connected components", "shape analysis", "Hu moments"]
---

# Segmentation & Contours

Segmentation splits images into regions, allowing you to isolate shapes, track outlines, and label connected components.

## Find Contours

Scans binary (thresholded) images to find shape boundaries, returning lists of connected coordinate points.

```rust
let contours = image.find_contours()?;
println!("Found {} contours", contours.len());
```

## Shape Analysis

### Convex Hull

Computes the convex boundary wrapping around a contour using Andrew's Monotone Chain algorithm.

```rust
let hull = contour.convex_hull();
```

### Area & Perimeter

```rust
let m = contour.moments();
let area = m.m00; // m00 represents the contour area
```

### Centroid

```rust
let m = contour.moments();
if let Some(centroid) = m.centroid() {
    println!("Center of mass: ({:.2}, {:.2})", centroid.x, centroid.y);
}
```

### Moments

Computes image moments representing spatial distribution.

```rust
let m = contour.moments();
println!("Area (m00): {}", m.m00);
println!("Centroid: {:?}", m.centroid());
```

### Hu Moments & Shape Matching

```rust
let hu = ShapeAnalysis::hu_moments(&m);
let diff = ShapeAnalysis::match_shapes(&m1, &m2);
```

## Connected Components with Stats

Labels separated foreground structures in a binary image and calculates statistics (bounding box, area, centroid).

```rust
let (labeled_mask, stats) = image.connected_components_with_stats()?;

for stat in stats {
    println!("Component ID: {}", stat.label);
    println!(" - Area: {} pixels", stat.area);
    println!(" - Centroid: {:?}", stat.centroid);
}
```

## Semantic Segmentation

```rust
let segmenter = Segmenter::<Backend>::from_onnx("model.onnx", &device)?;
let mask = segmenter.segment(&image)?;
// mask.mask is a Tensor<B, 2, Int> of shape [H, W] with class labels
```
