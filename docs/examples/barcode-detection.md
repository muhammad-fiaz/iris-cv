---
title: "Barcode Detection Example"
description: "Barcode detection and decoding with Iris."
keywords: ["barcode", "detection", "decoding"]
---

# Barcode Detection

Demonstrates barcode detection using the BarcodeDetector. Loads a real image and attempts to detect and decode barcodes.

```bash
cargo run --example barcode_detection --features wgpu
```

## Source

```rust
{{#include ../../../examples/barcode_detection.rs}}
```
