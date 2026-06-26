---
title: "Photo Processing Example"
description: "Non-Local Means denoising and Mertens exposure fusion with Iris."
keywords: ["denoising", "NLM", "exposure fusion", "Mertens"]
---

# Photo Processing

Demonstrates non-local means denoising and Mertens exposure fusion.

```bash
cargo run --example photo_processing --features wgpu
```

## Source

```rust
{{#include ../../../examples/photo_processing.rs}}
```
