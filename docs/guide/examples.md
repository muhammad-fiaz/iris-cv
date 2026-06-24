---
title: "Examples"
description: "Run complete examples — Canny edges, contours, QR detection, face recognition, filters, morphology, tracking, and more."
keywords: ["examples", "demos", "sample code", "tutorials", "canny", "contours", "QR detection"]
---

# Examples

The Iris repository includes several complete examples showcasing key capabilities of the library. You can run any example directly from the cloned repository.

## Running Examples

```bash
# Clone the repository
git clone https://github.com/muhammad-fiaz/iris.git
cd iris

# Run the Canny Edge Detection demo
cargo run --example canny

# Run the Contours and shapes demo
cargo run --example contours

# Run the QR Code reader demo
cargo run --example qr_detection

# Run the Face recognition demo
cargo run --example face_recognition

# Run the GUI window manager demo (requires wgpu backend)
cargo run --example gui_windows
```

## Available Examples

| Example | Command | Description |
|---|---|---|
| **Canny Edges** | `cargo run --example canny` | Reads an input image, computes gradients, runs non-maximum suppression, and saves edge maps. |
| **Contours** | `cargo run --example contours` | Finds connected components and traces outer boundaries, outputting polygon coordinates. |
| **GUI Windows** | `cargo run --example gui_windows` | Demonstrates the modern direct `Gui` API, registering callbacks, drawing trackbars, and rendering frames. |
| **QR Detection** | `cargo run --example qr_detection` | Scans frames to detect finder patterns, locating and decoding QR code payloads. |
| **Barcode Detection** | `cargo run --example barcode_detection` | Scans and decodes barcode payloads in images. |
| **Filters** | `cargo run --example filters` | Demonstrates blurs (box, Gaussian, median, bilateral, separable). |
| **Morphology** | `cargo run --example morphology` | Shows morphological dilation, erosion, opening, and closing operations. |
| **Threshold** | `cargo run --example threshold` | Demonstrates fixed and Otsu's thresholding methods. |
| **Tracking** | `cargo run --example tracking` | Demonstrates BackgroundSubtractor and KCF/CSRT trackers. |
| **Stitching** | `cargo run --example stitching` | Performs panorama stitching on overlapping images. |
| **Drawing** | `cargo run --example drawing` | Renders shapes and text labels on a canvas image. |
| **K-Means Clustering** | `cargo run --example kmeans_clustering` | Uses KMeans to cluster 2D coordinate points. |
| **Optical Flow** | `cargo run --example optical_flow` | Calculates dense Farneback flow and sparse LK tracking. |
| **Segmentation** | `cargo run --example segmentation` | Semantic segmentation and connected components labeling. |
| **OCR** | `cargo run --example ocr` | Performs optical character recognition on text regions. |
| **Video Processing** | `cargo run --example video_processing` | Mock capture and frame encoding. |
| **Photo Processing** | `cargo run --example photo_processing` | Non-Local Means denoising and exposure fusion. |
| **Safetensors** | `cargo run --example safetensors_loading` | Compiles weight loading structures to fetch and execute neural nets using `.safetensors`. |

