---
layout: home

hero:
  name: Iris
  text: Native Rust Computer Vision
  tagline: A fast computer vision library in pure Rust.
  image:
    src: /logo.svg
    alt: Iris Logo
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/muhammad-fiaz/iris

features:
  - icon: 🦀
    title: Pure idiomatic Rust
    details: Written natively in Rust from the ground up. Zero unsafe wrappers, zero external C/C++ dependencies.
  - icon: 🔥
    title: Burn Deep Learning
    details: Fully integrated with the Burn framework. Load ONNX, Safetensors, or native Burn model weights out of the box.
  - icon: ⚡
    title: GPU & WGPU Accelerated
    details: Harness hardware acceleration across WGPU, CUDA, Metal, and Vulkan automatically using compile-time features.
  - icon: 🧵
    title: Multi-Threaded CPU
    details: Automatic multi-threaded parallel execution across image filters, geometry warps, morphology, and drawing using Rayon.
  - icon: 🖥️
    title: Cross-Platform Direct UI
    details: Easy GUI window creation, mouse callbacks, trackbars, and rendering backends supported by Zed's GPUI framework.
  - icon: 🎯
    title: Type-Safe & Memory-Safe
    details: Built entirely in safe Rust — no unsafe code, no FFI bindings, no null pointers. Guaranteed memory safety and type safety at compile time.
  - icon: 🧩
    title: Advanced CV Algorithms
    details: Comprehensive set of computer vision algorithms including inpainting, stereo vision, Kalman filter, HOG descriptor, watershed segmentation, FLANN matcher, and mean-shift tracking.
---

::: warning NOTE
This project is still in active development. APIs and features may change before the first stable release.
:::

## Iris Documentation

Welcome to the official documentation for **Iris** — a pure-Rust computer vision library. Explore the guides below to get started:

### [Getting Started](/guide/getting-started)
Install Iris in your project, load an image, apply a Gaussian blur, detect edges with Canny, and save the result — all in a few lines of Rust.

### [Installation](/guide/installation)
Cargo features, backend options (WGPU, CUDA, LibTorch, Ndarray), and build configuration. Customize Iris for your hardware and use case.

### [Image Representation](/guide/image)
How images are represented as Burn tensors with shape `[Channels, Height, Width]`. Create, load, save, and query image properties.

### [Image Filters & Blur](/guide/filters)
Box blur, Gaussian blur, median filter, bilateral filter, and separable 2D filtering — all parallelized with Rayon or accelerated on GPU.

### [Edge Detection](/guide/edges)
Canny edge detection, Sobel, Scharr, and Laplacian gradient operators for structural analysis and feature extraction.

### [DNN & ONNX Inference](/guide/dnn)
Load and run ONNX, Safetensors, and PyTorch bin models. Preprocess inputs with `blob_from_image` and filter results with NMS.

### [API Reference](/api/)
Full reference for all types, modules, and functions — Image, Point, Rect, Scalar, filters, edges, morphology, DNN, and GUI.