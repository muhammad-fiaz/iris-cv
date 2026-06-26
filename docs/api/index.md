---
title: "API Reference"
description: "Overview of Iris's core API — Image, Point, Rect, Scalar, and primary modules for color, filters, edges, morphology, threshold, histogram, drawing, and more."
keywords: ["API reference", "documentation", "Image", "Point", "Rect", "Scalar", "modules"]
---

# API Reference Overview

Welcome to the API reference docs for **Iris**. This page introduces the major structures, traits, and modules exported by the library.

## Main Types

### Image
The primary image representation type. It wraps a multi-channel Burn tensor of shape `[Channels, Height, Width]`.
```rust
pub struct Image<B: Backend> {
    pub tensor: Tensor<B, 3>,
}
```

### Point
Represents a coordinate pair `(x, y)` in 2D space.
```rust
pub struct Point<T> {
    pub x: T,
    pub y: T,
}
```

### Rect
Defines a 2D bounding rectangle.
```rust
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}
```

### Scalar
A 4-element floating-point array, typically representing color values (e.g. RGBA).
```rust
pub struct Scalar(pub [f64; 4]);
```

### Size
A structure defining width and height dimensions.
```rust
pub struct Size<T> {
    pub width: T,
    pub height: T,
}
```

## Module Index

| Module | Description |
|---|---|
| **[core](/api/core)** | Point, Rect, Size, Scalar, Mat, Rng, image math ops, bitwise ops, normalization. |
| **[image](/api/image)** | Image struct, open, save, resize, crop, flip, rotate, geometric transforms. |
| **[filters](/api/filters)** | Box, Gaussian, median, bilateral blur, separable filter, distance transform, filter2D. |
| **[color](/api/color)** | RGB/HSV/HLS/XYZ/Lab/YUV/YCrCb/CMYK conversions, split/merge channels. |
| **[edges](/api/edges)** | Sobel, Canny, Scharr, Laplacian, Hough lines, Hough circles. |
| **[morphology](/api/morphology)** | Dilate, erode, open, close, morphology_ex, custom kernels, hit-or-miss, thinning, skeleton. |
| **[threshold](/api/threshold)** | Binary, Otsu, triangle, adaptive thresholding. |
| **[histogram](/api/histogram)** | Histogram computation, equalization, CLAHE, LUT, comparison, 2D histogram. |
| **[drawing](/api/drawing)** | Lines, rectangles, circles, ellipses, text, polylines, arrows, markers, fill poly. |
| **[noise](/api/noise)** | Gaussian, salt-and-pepper, and speckle noise generation. |
| **[contours](/api/contours)** | Contour detection, convex hull, moments, shape analysis, hierarchy, convexity defects. |
| **[features](/api/features)** | ORB feature detection, keypoints, BFMatcher, FLANN matcher, template matching. |
| **[tracking](/api/tracking)** | MOSSE tracker, MeanShift tracker, background subtraction. |
| **[dnn](/api/dnn)** | ONNX model loading, weight loaders, blob preprocessing, NMS. |
| **[video](/api/video)** | Video capture, reading, writing, frame iteration, metadata. |
| **[inpaint](/api/inpaint)** | Telea Fast Marching Method inpainting for damaged image regions. |
| **[stereo](/api/stereo)** | Stereo block matching for disparity map computation. |
| **[kalman](/api/kalman)** | Discrete Kalman filter for state estimation and prediction. |
| **[hog](/api/hog)** | Histogram of Oriented Gradients descriptor for feature extraction. |
| **[photo](/api/photo)** | Non-Local Means denoising and Mertens exposure fusion. |
| **[stitching](/api/stitching)** | Image stitching for panorama creation. |

## Importing the Prelude

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
```

The prelude re-exports all commonly used types: `Image`, `Point`, `Rect`, `Scalar`, `Size`, `Mat`, `Tensor`, `Backend`, `Result`, and all module-specific types.
