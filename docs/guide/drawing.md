---
title: "Drawing Canvas & Text Rendering"
description: "Draw lines, rectangles, circles, polygons, and render text on images using Iris's built-in 5x7 bitmap font."
keywords: ["drawing", "canvas", "text rendering", "shapes", "lines", "circles", "rectangles", "bitmap font"]
---

# Drawing Canvas & Text Rendering

Drawing shapes and rendering text onto frames is useful for displaying visualization boxes, showing tracking overlays, and labeling image outputs.

## Basic Geometric Shapes

All drawing functions operate directly on the `Image` struct (manipulating internal pixel bytes on the CPU and returning the modified frame):

### Lines
Draws a straight line from `p1` to `p2` with a specified `Scalar` color.

```rust
let p1 = Point::new(10, 10);
let p2 = Point::new(390, 10);
let color = Scalar::new(1.0, 0.0, 0.0, 0.0); // Red
let image = image.draw_line(p1, p2, color)?;
```

### Rectangles
Draws a rectangle with the specified borders or fills it.

```rust
let rect = Rect::new(50, 50, 100, 100);
let color = Scalar::new(0.0, 1.0, 0.0, 0.0); // Green
// Draw with thickness = 2
let image = image.draw_rectangle(rect, color, 2)?;

// Draw a filled rectangle with thickness = -1
let image = image.draw_rectangle(rect, color, -1)?;
```

### Circles
Draws a circle around a center point with a given radius.

```rust
let center = Point::new(200, 150);
let radius = 40;
let color = Scalar::new(0.0, 0.0, 1.0, 0.0); // Blue
// Draw a filled circle
let image = image.draw_circle(center, radius, color, -1)?;
```

## Rendering Text

Iris contains a lightweight, built-in **5x7 bitmap font** (`FONT_5X7`) to render standard ASCII labels onto images directly, eliminating the need to link external system font files.

```rust
let org = Point::new(50, 250);
let scale = 2;
let color = Scalar::all(1.0); // White text
let image = image.draw_text("Iris library", org, scale, color)?;
```
