---
title: "Examples"
description: "Run complete examples — Canny edges, contours, filters, morphology, drawing, tracking, optical flow, video processing, and more."
keywords: ["examples", "demos", "sample code", "tutorials", "canny", "contours", "filters"]
---

# Examples

The Iris repository includes several complete examples showcasing key capabilities of the library. You can run any example directly from the cloned repository.

## Running Examples

```bash
# Clone the repository
git clone https://github.com/muhammad-fiaz/iris.git
cd iris

# Run any example with the wgpu backend
cargo run --example <name> --features wgpu
```

## Available Examples

| Example | Command | Description |
|---|---|---|
| **Image Loading** | `cargo run --example image_loading --features wgpu` | Demonstrates image I/O operations. |
| **Canny Edges** | `cargo run --example canny --features wgpu` | Generates a test image with a high-contrast square, computes Canny edges, and saves the result. |
| **Filters** | `cargo run --example filters --features wgpu` | Demonstrates box, Gaussian, median, bilateral, and separable filters. |
| **Drawing** | `cargo run --example drawing --features wgpu` | Renders lines, rectangles, circles, and text labels on a canvas. |
| **Color Processing** | `cargo run --example color_processing --features wgpu` | Demonstrates RGB/HSV/HLS/YUV/Lab color space conversions. |
| **Image Utils** | `cargo run --example image_utils --features wgpu` | Utility functions for image manipulation. |
| **API Showcase** | `cargo run --example api_showcase --features wgpu` | Demonstrates color conversions, noise generation, filter2D, CLAHE, LUT, inRange, normalize, drawing extras, custom morph kernels, convert_scale_abs, and histogram comparison. |
| **Contours** | `cargo run --example contours --features wgpu` | Creates a binary image, finds contours, computes convex hull and moments. |
| **Morphology** | `cargo run --example morphology --features wgpu` | Shows dilation, erosion, opening, and closing with structuring elements. |
| **Threshold** | `cargo run --example threshold --features wgpu` | Demonstrates fixed, Otsu, and triangle thresholding methods. |
| **Optical Flow** | `cargo run --example optical_flow --features wgpu` | Calculates dense Farneback flow and sparse Lucas-Kanade tracking. |
| **Tracking** | `cargo run --example tracking --features wgpu` | Demonstrates BackgroundSubtractor and object tracking with MOSSE. |
| **QR Detection** | `cargo run --example qr_detection --features wgpu` | Scans frames to detect and decode QR code payloads. |
| **Face Recognition** | `cargo run --example face_recognition --features wgpu` | Detects faces and extracts/recognition face embeddings. |
| **ONNX Inference** | `cargo run --example onnx_inference --features wgpu` | Loads and runs an ONNX model for inference. |
| **Photo Processing** | `cargo run --example photo_processing --features wgpu` | Non-Local Means denoising and HDR exposure fusion. |
| **Segmentation** | `cargo run --example segmentation --features wgpu` | Semantic segmentation and connected components labeling. |
| **K-Means Clustering** | `cargo run --example kmeans_clustering --features wgpu` | Uses KMeans to cluster 2D coordinate points. |
| **Camera Calibration** | `cargo run --example camera_calibration --features wgpu` | Demonstrates camera matrix estimation and homography. |
| **ArUco Pose** | `cargo run --example aruco_pose --features wgpu` | ArUco marker detection and pose estimation. |
| **Barcode Detection** | `cargo run --example barcode_detection --features wgpu` | Scans and decodes barcode payloads in images. |
| **OCR** | `cargo run --example ocr --features wgpu` | Performs optical character recognition on text regions. |
| **GUI Windows** | `cargo run --example gui_windows --features wgpu` | Demonstrates the modern direct `Gui` API with trackbars and callbacks. |
| **Safetensors** | `cargo run --example safetensors_loading --features wgpu` | Compiles weight loading structures to fetch and execute neural nets. |
| **Stitching** | `cargo run --example stitching --features wgpu` | Performs panorama stitching on overlapping images. |
| **GIF & Video** | `cargo run --example gif_video --features wgpu` | GIF and video loading/processing. |
| **Image Sequence** | `cargo run --example image_sequence --features wgpu` | Loads and processes image sequences. |
| **Video Processing** | `cargo run --example video_processing --features wgpu` | Mock capture and frame encoding. |
| **Video Metadata** | `cargo run --example video_metadata --features wgpu` | Reads and displays video file metadata. |
| **Video Frame Ops** | `cargo run --example video_frame_ops --features wgpu` | Frame extraction, iteration, and processing. |
| **Generate Assets** | `cargo run --example generate_assets --features wgpu` | Generates test assets programmatically. |