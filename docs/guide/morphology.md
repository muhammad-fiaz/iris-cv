---
title: "Morphological Operations"
description: "Perform morphological image transformations — dilation, erosion, opening, closing, custom kernels, and structuring elements."
keywords: ["morphology", "dilation", "erosion", "opening", "closing", "structuring element", "binary image"]
---

# Morphological Operations

Morphological operations are shape-based image transformations that process images using structuring elements. They are commonly used on binary or grayscale images for filtering noise, separating touching elements, or detecting corners.

## Key Operations

### Dilation

Grows regions of foreground pixels, taking the local maximum value. It expands bright regions in the image.

```rust
// Apply dilation with a kernel size of 3
let dilated = image.dilate(3)?;
```

### Erosion

Shrinks regions of foreground pixels, taking the local minimum value. It expands dark regions in the image.

```rust
// Apply erosion with a kernel size of 3
let eroded = image.erode(3)?;
```

### Morphological Opening

Erosion followed by dilation. Useful for removing small bright noise objects.

```rust
let opened = image.morph_open(3)?;
```

### Morphological Closing

Dilation followed by erosion. Useful for filling small holes or joining segments.

```rust
let closed = image.morph_close(3)?;
```

### Morphology Ex

Provides advanced morphological transformations combining dilation and erosion.

- **`MorphOp::Opening`**: Erosion followed by dilation.
- **`MorphOp::Closing`**: Dilation followed by erosion.
- **`MorphOp::Gradient`**: Difference between dilation and erosion. Highlights boundaries.
- **`MorphOp::TopHat`**: Difference between input image and opening. Isolates bright features smaller than the kernel.
- **`MorphOp::BlackHat`**: Difference between closing and input image. Isolates dark features smaller than the kernel.

```rust
let opened = image.morphology_ex(MorphOp::Opening, 3)?;
let gradient = image.morphology_ex(MorphOp::Gradient, 3)?;
```

## Custom Kernels

### Dilate / Erode with Custom Kernel

Apply morphological operations with a user-defined binary kernel.

```rust
let kernel: Vec<&[u8]> = vec![
    &[0, 1, 0],
    &[1, 1, 1],
    &[0, 1, 0],
];
let dilated = image.dilate_with_kernel(&kernel)?;
let eroded = image.erode_with_kernel(&kernel)?;
```

## Structuring Elements

Structuring elements (or kernels) represent the spatial shapes used during filtering. Iris provides standard pre-built shape models:

- **`MorphShape::Rect`**: Flat box elements.
- **`MorphShape::Cross`**: Cross-shaped orthogonal lines.
- **`MorphShape::Ellipse`**: Elliptical kernel mask.

```rust
let size = Size::new(5, 5);
let element = Morphology::get_structuring_element(MorphShape::Ellipse, size);
```
