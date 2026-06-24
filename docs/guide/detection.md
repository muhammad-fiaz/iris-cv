---
title: "Object Detection & Recognition"
description: "Detect faces, recognize embeddings, and decode QR codes and barcodes using Iris's built-in detection pipelines."
keywords: ["object detection", "face detection", "face recognition", "QR code", "barcode", "embedding"]
---

# Object Detection & Recognition

Iris provides built-in pipelines for object detection, face recognition, and barcode/QR reading.

## Face Detection & Recognition

### Detection
Detects human faces, bounding boxes, confidence scores, and facial landmark coordinates (left/right eyes, nose, left/right mouth).

```rust
let detector = FaceDetector::<Backend>::default();
let faces = detector.detect(&image)?;

for face in faces {
    println!("Face found with confidence: {}", face.confidence);
    println!("Bounding box: {:?}", face.bbox);
}
```

### Recognition (Embeddings)
Extracts unique 512-dimension face embedding vectors and computes similarity metrics.

```rust
let recognizer = FaceRecognizer::<Backend>::default();

// Extract embeddings
let emb1 = recognizer.extract_embedding(&face_img1)?;
let emb2 = recognizer.extract_embedding(&face_img2)?;

// Compute similarity score (cosine distance)
let score = recognizer.compute_similarity(&emb1, &emb2)?;
println!("Face similarity: {}", score);
```

## QR & Barcode Detection

Detects and decodes payloads from QR codes and standard barcodes inside images.

### QR Code Detector

```rust
let qr_detector = QrDetector::new();
let qr_codes = qr_detector.detect_and_decode(&image)?;

for qr in qr_codes {
    println!("QR decoded: '{}'", qr.payload);
    println!("Corners: {:?}", qr.corners);
}
```

### Barcode Detector

```rust
let barcode_detector = BarcodeDetector::new();
let barcodes = barcode_detector.detect_and_decode(&image)?;

for bc in barcodes {
    println!("Barcode payload: '{}' (format: {})", bc.payload, bc.format);
}
```
