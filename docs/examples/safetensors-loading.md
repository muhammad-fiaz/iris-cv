---
title: "SafeTensors Loading Example"
description: "Weight loading from binary and safetensors files, and NMS with Iris."
keywords: ["safetensors", "weight loading", "NMS"]
---

# SafeTensors Loading

Demonstrates weight loading from binary and safetensors files, and Non-Maximum Suppression for bounding box filtering.

```bash
cargo run --example safetensors_loading --features wgpu
```

## Source

```rust
{{#include ../../../examples/safetensors_loading.rs}}
```
