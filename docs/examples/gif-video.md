---
title: "GIF Video Example"
description: "GIF creation, reading, and roundtrip verification with Iris."
keywords: ["GIF", "video", "animation"]
---

# GIF Video

Demonstrates GIF creation and reading: generates animated frames, writes to GIF, reads back, and verifies frame data.

```bash
cargo run --example gif_video --features wgpu
```

## Source

```rust
{{#include ../../../examples/gif_video.rs}}
```
