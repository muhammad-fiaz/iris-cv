---
title: "Introduction to Iris"
description: "Introduction to Iris — a pure-Rust computer vision library that replaces OpenCV with zero C dependencies, GPU acceleration, and multi-threaded CPU support."
keywords: ["introduction", "why iris", "opencv alternative", "pure rust cv", "zero dependencies"]
---

# Introduction

Welcome to **Iris**! This library provides high-performance computer vision algorithms, image manipulation tools, and deep learning model hosting fully in safe Rust.

## Why Iris?

Traditional computer vision libraries (such as OpenCV) are large, complex C++ codebases. Using them in Rust requires binding wrappers like `opencv-rust`, which introduce several development pain points:
- Complicated build toolchains, especially on Windows or bare-metal environments.
- Linking conflicts with system-level dynamic libraries.
- Unsafe memory pointers that bypass Rust's borrow checker.

**Iris** solves these issues by writing all algorithms natively in Rust:
- **Zero-Dependency Compilation**: Clean Cargo builds with no external C dependencies.
- **Modern Hardware Acceleration**: Thanks to Burn, you can run the exact same image operations on a CPU, WGPU (Vulkan, DX12, Metal), or CUDA simply by switching the backend type parameter.
- **Rayon Multi-Threading**: When running on the CPU, heavy loops (like bilateral filter and perspective warps) scale automatically to all available CPU cores.
