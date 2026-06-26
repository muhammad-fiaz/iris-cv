---
title: "Image Loading Example"
description: "Load, resize, annotate, and save images with Iris."
keywords: ["image loading", "image save", "resize", "annotate"]
---

# Image Loading

Loads a real image, resizes it, adds annotations, and saves the result.

```bash
cargo run --example image_loading --features wgpu
```

## Source

```rust
{{#include ../../../examples/image_loading.rs}}
```
