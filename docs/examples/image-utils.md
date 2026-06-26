---
title: "Image Utils Example"
description: "Utility operations — filter2D, blending, LUT, noise, histogram comparison with Iris."
keywords: ["image utilities", "filter2D", "blending", "LUT"]
---

# Image Utils

Demonstrates image utility functions: filter2D, alpha blending, convert_scale_abs, copy_to with mask, normalize, LUT, noise generation, and histogram comparison.

```bash
cargo run --example image_utils --features wgpu
```

## Source

```rust
{{#include ../../../examples/image_utils.rs}}
```
