---
title: "OCR Example"
description: "OCR text recognition pipeline with Iris."
keywords: ["OCR", "text recognition", "text detection"]
---

# OCR

Demonstrates OCR text recognition using the OcrPipeline.

```bash
cargo run --example ocr --features wgpu
```

## Source

```rust
// Demonstrates OCR text recognition using the OcrPipeline.
// Loads the test pattern image and runs text detection.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load a real image (test pattern may not contain text, but demonstrates pipeline)
    let img: Image<Backend> = Image::open("assets/images/test_pattern.png", &device)?;
    println!(
        "Loaded image: {}x{} ({} channels)",
        img.width(),
        img.height(),
        img.channels()
    );

    // Run OCR text recognition
    println!("Running OCR text recognition...");
    let ocr = OcrPipeline::default();
    let ocr_results = ocr.recognize(&img)?;

    println!("OCR Results:");
    for result in &ocr_results {
        println!(
            "  - Text: '{}', bbox: {:?}, confidence: {:.4}",
            result.text, result.bbox, result.confidence
        );
    }

    if ocr_results.is_empty() {
        println!("  (No text detected in test pattern — expected)");
    }

    img.save("output_ocr.png")?;
    println!("Saved input image to 'output_ocr.png'");

    Ok(())
}
```
