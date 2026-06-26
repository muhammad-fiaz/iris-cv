---
title: "Image Sequence Example"
description: "Load numbered image sequences into frames with Iris."
keywords: ["image sequence", "frame loading"]
---

# Image Sequence

Demonstrates loading a sequence of numbered PNG images into Frame objects, creating a FrameIterator, and seeking to specific frames.

```bash
cargo run --example image_sequence --features wgpu
```

## Source

```rust
{{#include ../../../examples/image_sequence.rs}}
```
