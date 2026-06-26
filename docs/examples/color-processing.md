---
title: "Color Processing Example"
description: "Color space conversions, CLAHE, histogram equalization, and in-range thresholding with Iris."
keywords: ["color processing", "color spaces", "CLAHE", "histogram"]
---

# Color Processing

Demonstrates color space conversions (HSV, LAB, YUV, YCrCb, HLS, XYZ), CLAHE, histogram equalization, and in-range color thresholding.

```bash
cargo run --example color_processing --features wgpu
```

## Source

```rust
{{#include ../../../examples/color_processing.rs}}
```
