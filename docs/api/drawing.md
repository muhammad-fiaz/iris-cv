---
title: "Drawing Module Reference"
description: "API reference for Iris drawing module — draw_line, draw_rectangle, draw_circle, draw_ellipse, draw_text, draw_polyline, fill_poly, draw_arrowed_line, draw_marker."
keywords: ["drawing", "shapes", "lines", "rectangles", "circles", "ellipses", "text", "polyline", "polygon", "marker", "arrow"]
---

# Drawing Module Reference

Provides drawing primitives for rendering shapes and text on images.

## Drawing Operations

```rust
impl<B: Backend> Image<B> {
    pub fn draw_line(self, p1: Point<usize>, p2: Point<usize>, color: Scalar) -> Result<Self>;
    pub fn draw_rectangle(self, rect: Rect<usize>, color: Scalar, thickness: i32) -> Result<Self>;
    pub fn draw_circle(self, center: Point<usize>, radius: usize, color: Scalar, thickness: i32) -> Result<Self>;
    pub fn draw_ellipse(
        self,
        center: Point<usize>,
        axes: (usize, usize),
        angle: f32,
        start_angle: f32,
        end_angle: f32,
        color: Scalar,
        thickness: i32,
    ) -> Result<Self>;
    pub fn draw_text(self, text: &str, org: Point<usize>, scale: usize, color: Scalar) -> Result<Self>;
    pub fn draw_polyline(self, points: &[Point<usize>], color: Scalar, thickness: i32) -> Result<Self>;
    pub fn fill_poly(self, points: &[Point<usize>], color: Scalar) -> Result<Self>;
    pub fn draw_arrowed_line(
        self,
        p1: Point<usize>,
        p2: Point<usize>,
        color: Scalar,
        thickness: i32,
        tip_length: f32,
    ) -> Result<Self>;
    pub fn draw_marker(
        self,
        center: Point<usize>,
        color: Scalar,
        marker_type: MarkerType,
        marker_size: usize,
    ) -> Result<Self>;
}
```

## Marker Types

```rust
pub enum MarkerType {
    Cross,
    TiltedCross,
    Diamond,
    Square,
    Circle,
    Filled,
}
```

## Notes

- Use `thickness = -1` to draw filled shapes (rectangles, circles, ellipses).
- Text rendering uses the built-in 5x7 bitmap font — no external font files required.
- Colors are specified as `Scalar::new(r, g, b, a)` with values in `[0.0, 1.0]`.

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();

// Create black canvas
let canvas = Image::<Backend>::zeros(3, 400, 300, &device)?;

// Draw shapes
let canvas = canvas
    .draw_line(Point::new(10, 10), Point::new(390, 10), Scalar::new(1.0, 0.0, 0.0, 0.0))?
    .draw_rectangle(Rect::new(50, 50, 100, 100), Scalar::new(0.0, 1.0, 0.0, 0.0), 2)?
    .draw_circle(Point::new(300, 100), 40, Scalar::new(0.0, 0.0, 1.0, 0.0), -1)?
    .draw_text("Iris library", Point::new(50, 200), 2, Scalar::all(1.0))?;

canvas.save("output.png")?;
```
