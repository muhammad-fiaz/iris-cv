---
title: "Canny Edge Detection Example"
description: "Canny edge detection on real images with Iris."
keywords: ["canny", "edge detection"]
---

# Canny Edge Detection

Loads the test pattern image and applies Canny edge detection with two threshold levels.

```bash
cargo run --example canny --features wgpu
```

## Source

```rust
{{#include ../../../examples/canny.rs}}
```
