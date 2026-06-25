---
title: "What is Iris?"
description: "Iris is a fast, pure-Rust computer vision library powered by Burn. Learn about its goals, architecture, and modular design."
keywords: ["Iris", "what is iris", "rust computer vision", "burn framework", "CV library"]
---

# What is Iris?

**Iris** is a fast computer vision library written entirely in pure Rust. It leverages the **Burn** framework for GPU/CPU tensor computations — memory safe, type safe, with no external C/C++ dependencies.

## Key Goals

1. **Pure Rust Ecosystem**: Keep dependencies clean, safe, and fully compiled under the Cargo toolchain.
2. **Familiar CV API**: Provide developers with a clean, modern API covering Image I/O, Filters, Gradients, Contour tracking, Cameras, Drawing, and Detection.
3. **Deep Learning First**: Out-of-the-box loading of ONNX and Safetensors models, powered by Burn backends (WGPU, CUDA, Metal, Ndarray).
4. **Multi-Threaded CPU Acceleration**: All computationally intensive CPU operations are parallelized using Rayon, allowing high throughput when not running on a GPU.

## Project Structure

Iris is designed with modularity in mind:

| Module | Description |
|---|---|
| **`core`** | Basic matrix representations (`Mat`), geometric types (`Point`, `Rect`, `Size`, `Scalar`), and RNG utilities. |
| **`image`** | Image struct, file I/O, resizing, cropping, flipping, rotation, warping, and remapping. |
| **`filters`** | Box blur, Gaussian blur, median filter, bilateral filter, separable 2D filtering, distance transform, and more. |
| **`edges`** | Canny edge detection, Sobel, Scharr, Laplacian, Hough lines, and Hough circles. |
| **`morphology`** | Dilation, erosion, opening, closing, and custom structuring elements. |
| **`threshold`** | Binary, Otsu, triangle, and adaptive thresholding. |
| **`color`** | RGB/HSV/HLS/XYZ/Lab/YUV/YCrCb conversions, channel splitting and merging. |
| **`histogram`** | Histogram computation, equalization, CLAHE, LUT, and comparison. |
| **`drawing`** | Lines, rectangles, circles, ellipses, text, polylines, arrows, and markers. |
| **`contours`** | Contour detection, convex hull, moments, shape analysis, and matching. |
| **`features`** | ORB feature detection (FAST + BRIEF), keypoints, and BFMatcher. |
| **`optical_flow`** | Lucas-Kanade sparse flow and Farneback dense flow. |
| **`tracking`** | MOSSE tracker, background subtraction, and mean-shift tracking. |
| **`photo`** | Non-Local Means denoising and HDR merge (Mertens). |
| **`dnn`** | ONNX model loading, weight loaders, blob preprocessing, and NMS. |
| **`segmentation`** | Semantic segmentation, connected components with stats, and watershed segmentation. |
| **`noise`** | Gaussian, salt-and-pepper, and speckle noise generation. |
| **`camera`** | Camera capture and camera calibration. |
| **`face`** | Face detection and face recognition. |
| **`video`** | Video capture, reading, writing, frame iteration, and metadata. |
| **`ml`** | K-Means clustering. |
| **`inpainting`** | Image inpainting for restoring damaged or missing regions. |
| **`stereo`** | Stereo vision and depth estimation. |
| **`kalman`** | Kalman filter for state estimation and tracking. |
| **`hog`** | Histogram of Oriented Gradients (HOG) descriptor for object detection. |
| **`feature_matching`** | FLANN-based approximate nearest neighbor matching for feature descriptors. |