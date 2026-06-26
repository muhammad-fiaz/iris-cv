---
title: "Examples"
description: "Run complete examples — Canny edges, contours, filters, morphology, drawing, tracking, optical flow, video processing, and more."
keywords: ["examples", "demos", "sample code", "tutorials", "canny", "contours", "filters"]
---

# Examples

The Iris repository includes several complete examples showcasing key capabilities of the library. Browse the [Examples Overview](/examples/) for full source code and descriptions.

## Running Examples

```bash
# Clone the repository
git clone https://github.com/muhammad-fiaz/iris-cv.git
cd iris-cv

# Run any example with the wgpu backend
cargo run --example <name> --features wgpu
```

::: tip
Generate test assets first: `cargo run --example generate_assets --features wgpu`
:::

## Available Examples

### Image Processing

| Example | Command | Description |
|---|---|---|
| [Image Loading](/examples/image-loading) | `cargo run --example image_loading --features wgpu` | Load, resize, annotate, and save images |
| [Filters](/examples/filters) | `cargo run --example filters --features wgpu` | Box, Gaussian, median, bilateral, and separable filters |
| [Canny](/examples/canny) | `cargo run --example canny --features wgpu` | Canny edge detection on real images |
| [Morphology](/examples/morphology) | `cargo run --example morphology --features wgpu` | Dilation, erosion, opening, closing |
| [Threshold](/examples/threshold) | `cargo run --example threshold --features wgpu` | Binary, truncation, to-zero, Otsu thresholding |
| [Contours](/examples/contours) | `cargo run --example contours --features wgpu` | Contour detection, convex hull, moments |
| [Drawing](/examples/drawing) | `cargo run --example drawing --features wgpu` | Lines, rectangles, circles, text |

### Color & Photo

| Example | Command | Description |
|---|---|---|
| [Color Processing](/examples/color-processing) | `cargo run --example color_processing --features wgpu` | Color space conversions, CLAHE, in-range |
| [Photo Processing](/examples/photo-processing) | `cargo run --example photo_processing --features wgpu` | NLM denoising and Mertens exposure fusion |

### Features & Tracking

| Example | Command | Description |
|---|---|---|
| [Optical Flow](/examples/optical-flow) | `cargo run --example optical_flow --features wgpu` | Dense (Farneback) and sparse (Lucas-Kanade) flow |
| [Tracking](/examples/tracking) | `cargo run --example tracking --features wgpu` | Background subtraction and CSRT tracking |

### Video

| Example | Command | Description |
|---|---|---|
| [GIF Video](/examples/gif-video) | `cargo run --example gif_video --features wgpu` | GIF creation, reading, roundtrip |
| [Video Processing](/examples/video-processing) | `cargo run --example video_processing --features wgpu` | Video writing and frame iteration |
| [Video Frame Ops](/examples/video-frame-ops) | `cargo run --example video_frame_ops --features wgpu` | Frame differencing, motion, batch, seeking |
| [Video Metadata](/examples/video-metadata) | `cargo run --example video_metadata --features wgpu` | Format detection, stream info, pixel formats |
| [Image Sequence](/examples/image-sequence) | `cargo run --example image_sequence --features wgpu` | Load numbered images into frames |

### Detection & Recognition

| Example | Command | Description |
|---|---|---|
| [Barcode Detection](/examples/barcode-detection) | `cargo run --example barcode_detection --features wgpu` | Barcode detection and decoding |
| [QR Detection](/examples/qr-detection) | `cargo run --example qr_detection --features wgpu` | QR code detection and decoding |
| [ArUco Pose](/examples/aruco-pose) | `cargo run --example aruco_pose --features wgpu` | ArUco marker detection and pose estimation |
| [Face Recognition](/examples/face-recognition) | `cargo run --example face_recognition --features wgpu` | Face detection and recognition embeddings |
| [Segmentation](/examples/segmentation) | `cargo run --example segmentation --features wgpu` | Semantic segmentation and connected components |

### DNN & ML

| Example | Command | Description |
|---|---|---|
| [ONNX Inference](/examples/onnx-inference) | `cargo run --example onnx_inference --features wgpu` | ONNX model loading and inference |
| [SafeTensors Loading](/examples/safetensors-loading) | `cargo run --example safetensors_loading --features wgpu` | Weight loading and NMS |
| [K-Means Clustering](/examples/kmeans-clustering) | `cargo run --example kmeans_clustering --features wgpu` | K-Means clustering on 2D points |
| [OCR](/examples/ocr) | `cargo run --example ocr --features wgpu` | OCR text recognition pipeline |

### Advanced

| Example | Command | Description |
|---|---|---|
| [Stitching](/examples/stitching) | `cargo run --example stitching --features wgpu` | Image panorama stitching |
| [Camera Calibration](/examples/camera-calibration) | `cargo run --example camera_calibration --features wgpu` | Camera calibration, projection, homography |
| [API Showcase](/examples/api-showcase) | `cargo run --example api_showcase --features wgpu` | Core API overview |
| [Image Utils](/examples/image-utils) | `cargo run --example image_utils --features wgpu` | Utility ops (filter2D, blend, LUT, noise) |
| [GUI Windows](/examples/gui-windows) | `cargo run --example gui_windows --features wgpu` | GUI windows, trackbars, mouse callbacks |
| [Generate Assets](/examples/generate-asset) | `cargo run --example generate_assets --features wgpu` | Generate test images for examples |
