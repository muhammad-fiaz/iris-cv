---
title: "QR Detection Example"
description: "QR code detection and decoding with Iris."
keywords: ["QR code", "detection", "decoding"]
---

# QR Detection

Demonstrates QR code detection and decoding using QrDetector.

```bash
cargo run --example qr_detection --features wgpu
```

## Source

```rust
// Demonstrates QR code detection and decoding using QrDetector.
// Loads a real image and searches for QR codes.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load a real image to search for QR codes
    let image: Image<Backend> = Image::open("assets/images/test_pattern.png", &device)?;

    // Detect QR codes
    let detector = QrDetector::new();
    let qr_codes = detector.detect_and_decode(&image)?;

    println!("Detected {} QR code(s):", qr_codes.len());
    for qr in &qr_codes {
        println!("  - Payload: '{}'", qr.payload);
        println!("    Corners: {:?}", qr.corners);
    }

    if qr_codes.is_empty() {
        println!("  (No QR codes found in test_pattern — expected)");
    }

    image.save("output_qr_detection.png")?;
    println!("Saved input image to 'output_qr_detection.png'");

    Ok(())
}
```
