---
title: "ONNX Inference Example"
description: "ONNX model loading and inference with Iris."
keywords: ["ONNX", "inference", "model loading"]
---

# ONNX Inference

Demonstrates ONNX model loading, preprocessing, and raw prediction.

```bash
cargo run --example onnx_inference --features wgpu
```

## Source

```rust
{{#include ../../../examples/onnx_inference.rs}}
```
