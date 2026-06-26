---
title: "Face Recognition Example"
description: "Face detection and recognition embeddings with Iris."
keywords: ["face detection", "face recognition", "embeddings"]
---

# Face Recognition

Demonstrates face detection and face recognition embedding extraction using mock models.

```bash
cargo run --example face_recognition --features wgpu
```

## Source

```rust
// Demonstrates face detection and face recognition embedding extraction.
// Uses mock models (no real ONNX weights required).

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load a real image for face detection
    let img1: Image<Backend> = Image::open("assets/images/test_pattern.png", &device)?;

    // 1. Face detection
    let detector = FaceDetector::<Backend>::default();
    let faces = detector.detect(&img1)?;
    println!("Detected {} face(s) in image 1", faces.len());

    // 2. Face recognition with mock embedding extraction
    let recognizer = FaceRecognizer::from_onnx("facenet_mock.onnx", &device)?;

    let emb1 = recognizer.extract_embedding(&img1)?;
    println!("Embedding 1 shape: {:?}", emb1.dims());

    // Create a second image from gradient for comparison
    let img2: Image<Backend> = Image::open("assets/images/gradient.png", &device)?;
    let emb2 = recognizer.extract_embedding(&img2)?;
    println!("Embedding 2 shape: {:?}", emb2.dims());

    // 3. Compute cosine similarity
    let similarity = recognizer.compute_similarity(&emb1, &emb2)?;
    println!("Face similarity score: {:.4}", similarity);

    Ok(())
}
```
