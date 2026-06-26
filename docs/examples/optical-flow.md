---
title: "Optical Flow Example"
description: "Dense (Farneback) and sparse (Lucas-Kanade) optical flow with Iris."
keywords: ["optical flow", "Farneback", "Lucas-Kanade", "motion"]
---

# Optical Flow

Demonstrates dense and sparse optical flow computation on two real images.

```bash
cargo run --example optical_flow --features wgpu
```

## Source

```rust
{{#include ../../../examples/optical_flow.rs}}
```
