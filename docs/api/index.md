---
title: "API Reference"
description: "Overview of Iris's core API — Image, Point, Rect, Scalar, and primary modules for filters, edges, morphology, DNN, and GUI."
keywords: ["API reference", "documentation", "Image", "Point", "Rect", "Scalar", "modules"]
---

# API Reference Overview

Welcome to the API reference docs for **Iris**. This page introduces the major structures, traits, and modules exported by the library.

## Main Types

### Image
The primary image representation type. It wraps a multi-channel Burn tensor of shape `[Channels, Height, Width]`.
```rust
pub struct Image<B: Backend> {
    pub tensor: Tensor<B, 3>,
}
```

### Point
Represents a coordinate pair `(x, y)` in 2D space.
```rust
pub struct Point<T> {
    pub x: T,
    pub y: T,
}
```

### Rect
Defines a 2D bounding rectangle.
```rust
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}
```

### Scalar
A 4-element floating-point array, typically representing color values (e.g. RGBA).
```rust
pub struct Scalar(pub [f64; 4]);
```

## Primary Modules

- **`filters`**: Functions to blur, smooth, and apply custom separable convolutions.
- **`edges`**: Gradients, Laplacian, Sobel, and Canny filters.
- **`morphology`**: Dilation, erosion, opening, closing, and advanced structuring elements.
- **`dnn`**: Preprocessing inputs, ONNX model loading, and non-maximum suppression.
- **`gui`**: Create named windows, display matrices, and handle slider trackbars.
