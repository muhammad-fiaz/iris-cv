---
title: "API Showcase Example"
description: "Core API overview — color, noise, filters, drawing, morphology with Iris."
keywords: ["API showcase", "overview", "features"]
---

# API Showcase

Demonstrates the core API features: color space conversions, noise generation, custom convolution filters, CLAHE, LUT, in_range, normalize, drawing, and morphological kernels.

```bash
cargo run --example api_showcase --features wgpu
```

## Source

```rust
{{#include ../../../examples/api_showcase.rs}}
```
