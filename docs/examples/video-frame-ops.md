---
title: "Video Frame Operations Example"
description: "Frame differencing, motion magnitudes, batch tensor, seeking, and looping with Iris."
keywords: ["video", "frame operations", "motion", "batch"]
---

# Video Frame Operations

Demonstrates frame differencing, motion magnitudes, batch tensor conversion, random access, seeking, and looping.

```bash
cargo run --example video_frame_ops --features wgpu
```

## Source

```rust
{{#include ../../../examples/video_frame_ops.rs}}
```
