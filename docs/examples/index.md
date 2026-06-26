---
title: "Examples"
description: "Complete collection of Iris CV examples — image loading, filters, edge detection, color processing, video, DNN inference, and more."
keywords: ["examples", "tutorials", "code samples", "demo"]
---

# Examples

Run any example with:

```bash
cargo run --example <name> --features wgpu
```

::: note
All examples require the `wgpu` feature. Generate test assets first with `cargo run --example generate_assets --features wgpu`.
:::

## Image Processing

| Example | Description |
|---|---|
| [image_loading](/examples/image-loading) | Load, resize, annotate, and save images |
| [filters](/examples/filters) | Box, Gaussian, median, bilateral, and separable filters |
| [canny](/examples/canny) | Canny edge detection on real images |
| [morphology](/examples/morphology) | Dilation, erosion, opening, closing |
| [threshold](/examples/threshold) | Binary, truncation, to-zero, and Otsu thresholding |
| [contours](/examples/contours) | Contour detection, convex hull, moments |
| [drawing](/examples/drawing) | Draw lines, rectangles, circles, and text |

## Color & Histogram

| Example | Description |
|---|---|
| [color_processing](/examples/color-processing) | Color space conversions, CLAHE, in-range thresholding |
| [photo_processing](/examples/photo-processing) | NLM denoising and Mertens exposure fusion |

## Features & Tracking

| Example | Description |
|---|---|
| [optical_flow](/examples/optical-flow) | Dense (Farneback) and sparse (Lucas-Kanade) optical flow |
| [tracking](/examples/tracking) | Background subtraction and CSRT object tracking |

## Video

| Example | Description |
|---|---|
| [gif_video](/examples/gif-video) | GIF creation, reading, and roundtrip verification |
| [video_processing](/examples/video-processing) | Video writing and frame iteration |
| [video_frame_ops](/examples/video-frame-ops) | Frame differencing, motion, batch, seeking, looping |
| [video_metadata](/examples/video-metadata) | Format detection, stream info, pixel formats |
| [image_sequence](/examples/image-sequence) | Load numbered image sequences into frames |

## Detection & Recognition

| Example | Description |
|---|---|
| [barcode_detection](/examples/barcode-detection) | Barcode detection and decoding |
| [qr_detection](/examples/qr-detection) | QR code detection and decoding |
| [aruco_pose](/examples/aruco-pose) | ArUco marker detection and pose estimation |
| [face_recognition](/examples/face-recognition) | Face detection and recognition embeddings |
| [segmentation](/examples/segmentation) | Semantic segmentation and connected components |

## DNN & ML

| Example | Description |
|---|---|
| [onnx_inference](/examples/onnx-inference) | ONNX model loading and inference |
| [safetensors_loading](/examples/safetensors-loading) | Weight loading and NMS |
| [kmeans_clustering](/examples/kmeans-clustering) | K-Means clustering on 2D points |
| [ocr](/examples/ocr) | OCR text recognition pipeline |

## Advanced

| Example | Description |
|---|---|
| [stitching](/examples/stitching) | Image panorama stitching |
| [camera_calibration](/examples/camera-calibration) | Camera calibration, projection, homography |
| [api_showcase](/examples/api-showcase) | Core API overview (color, noise, filters, drawing) |
| [image_utils](/examples/image-utils) | Utility ops (filter2D, blend, LUT, noise, histogram) |
| [gui_windows](/examples/gui-windows) | GUI windows, trackbars, mouse callbacks |
| [generate_assets](/examples/generate-asset) | Generate test images for examples |
