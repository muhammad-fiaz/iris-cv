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
let area = contour.contour_area();
let perimeter = contour.arc_length(true); // true = closed loop
```

### Bounding Rectangles

```rust
// Straight bounding box
let rect = contour.bounding_rect();

// Minimum area rotated rectangle
let (center, size, angle) = contour.min_area_rect();
```

### Hu Moments & Shape Matching
Computes 7 scale/rotation/translation invariant Hu Moments to evaluate shape similarities.

```rust
let m = contour.moments();
let hu_moments = ShapeAnalysis::hu_moments(&m);

// Compare two shape moments (returns distance score)
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
