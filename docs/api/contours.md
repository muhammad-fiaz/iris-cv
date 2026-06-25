---
title: "Contours Module Reference"
description: "API reference for Iris contours module — find_contours, convex_hull, convexity_defects, contour hierarchy, moments, arc_length, contour_area, approx_poly_dp, bounding_rect, min_area_rect, hu_moments, match_shapes."
keywords: ["contours", "convex hull", "moments", "shape analysis", "bounding rectangle", "arc length", "contour area", "convexity defects", "contour hierarchy", "RetrievalMode"]
---

# Contours Module Reference

Provides contour detection, geometric analysis, and shape matching.

## Contour Finding

```rust
impl<B: Backend> Image<B> {
    pub fn find_contours(&self) -> Result<Vec<Contour>>;
}
```

## Contour Struct

```rust
pub struct Contour {
    pub points: Vec<Point<usize>>,
}

impl Contour {
    pub fn new(points: Vec<Point<usize>>) -> Self;
    pub fn convex_hull(&self) -> Self;
    pub fn moments(&self) -> Moments;
}
```

## Moments

```rust
pub struct Moments {
    pub m00: f64,
    pub m10: f64,
    pub m01: f64,
    pub m20: f64,
    pub m02: f64,
    pub m11: f64,
    pub m30: f64,
    pub m03: f64,
    pub m21: f64,
    pub m12: f64,
}

impl Moments {
    pub fn centroid(&self) -> Option<Point<f64>>;
}
```

## Shape Analysis

```rust
pub struct ShapeAnalysis;

impl ShapeAnalysis {
    pub fn hu_moments(m: &Moments) -> [f64; 7];
    pub fn match_shapes(m1: &Moments, m2: &Moments) -> f64;
}
```

## Rotated Rectangle

```rust
pub struct RotatedRect {
    pub center: Point<f64>,
    pub size: Size<f64>,
    pub angle: f64,
}
```

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();
let img = Image::<Backend>::open("input.jpg", &device)?;

let contours = img.find_contours()?;
println!("Found {} contours", contours.len());

for contour in &contours {
    let hull = contour.convex_hull();
    let m = contour.moments();
    if let Some(centroid) = m.centroid() {
        println!("Centroid: ({:.1}, {:.1}), Area: {:.1}", centroid.x, centroid.y, m.m00);
    }
}

// Shape matching
if contours.len() >= 2 {
    let m1 = contours[0].moments();
    let m2 = contours[1].moments();
    let diff = ShapeAnalysis::match_shapes(&m1, &m2);
    println!("Shape difference: {:.4}", diff);
}
```

## Convexity Defects

```rust
pub struct ConvexityDefect {
    pub start: Point<usize>,
    pub end: Point<usize>,
    pub farthest: Point<usize>,
    pub distance: f64,
}

impl Contour {
    pub fn convexity_defects(&self) -> Vec<ConvexityDefect>;
}
```

Finds the convexity defects of a contour — points on the contour that lie beyond the convex hull. Each defect records the start/end points of the hull segment and the farthest contour point.

| Field | Description |
|---|---|
| `start` | Start point of the convex hull edge. |
| `end` | End point of the convex hull edge. |
| `farthest` | Point on the contour farthest from the hull edge. |
| `distance` | Perpendicular distance from `farthest` to the hull edge. |

### Example

```rust,ignore
let contour = &contours[0];
let defects = contour.convexity_defects();
for d in &defects {
    println!("Defect: farthest=({},{}), depth={:.2}", d.farthest.x, d.farthest.y, d.distance);
}
```

## Contour Hierarchy

```rust
pub enum RetrievalMode {
    External,
    List,
    CComp,
    Tree,
    FloodFill,
}

impl<B: Backend> Image<B> {
    pub fn find_contours_with_hierarchy(&self, mode: RetrievalMode) -> Result<(Vec<Contour>, Vec<[i32; 3]>)>;
}
```

Returns contours along with their parent-child hierarchy. Each hierarchy entry is `[next, previous, child]` — indices into the contour list (or `-1` if absent).

| Mode | Description |
|---|---|
| `RetrievalMode::External` | Retrieves only the extreme outer contours. |
| `RetrievalMode::List` | Retrieves all contours without hierarchy. |
| `RetrievalMode::CComp` | Retrieves all contours into a 2-level hierarchy (external and holes). |
| `RetrievalMode::Tree` | Retrieves all contours into a full tree hierarchy. |
| `RetrievalMode::FloodFill` | Same as Tree but does not connect contours at same level. |

### Example

```rust,ignore
let (contours, hierarchy) = img.find_contours_with_hierarchy(RetrievalMode::Tree)?;
for (i, h) in hierarchy.iter().enumerate() {
    println!("Contour {}: next={}, prev={}, child={}", i, h[0], h[1], h[2]);
}
```
