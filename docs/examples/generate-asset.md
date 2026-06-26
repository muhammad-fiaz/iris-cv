---
title: "Generate Assets Example"
description: "Generate test images for examples."
keywords: ["assets", "test images", "generation"]
---

# Generate Assets

Generates sample assets for Iris CV library examples: gradient, checkerboard, test pattern, grayscale gradient, and color bars images.

```bash
cargo run --example generate_assets --features wgpu
```

## Source

```rust
{{#include ../../../examples/generate_assets.rs}}
```
