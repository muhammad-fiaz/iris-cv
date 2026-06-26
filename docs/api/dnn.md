---
title: "DNN Module Reference"
description: "API reference for Iris DNN module — ONNX model loading, weight loaders, blob preprocessing, and NMS."
keywords: ["DNN", "ONNX", "model loading", "weight loader", "safetensors", "NMS", "blob"]
---

# DNN Module Reference

Contains definitions for neural network loaders, non-maximum suppression, and blob preprocessing helper pipelines.

## Model Interface

### `OnnxModel`

```rust
pub struct OnnxModel<B: Backend> {
    pub model_path: String,
    device: B::Device,
}

impl<B: Backend> OnnxModel<B> {
    pub fn load(path: impl AsRef<Path>, device: &B::Device) -> Result<Self>;
    pub fn predict_raw<const D1: usize, const D2: usize>(&self, input: Tensor<B, D1>) -> Result<Tensor<B, D2>>;
    pub fn preprocess(&self, image: &Image<B>) -> Result<Tensor<B, 4>>;
}
```

### Direct Builders

```rust
pub fn read_net<B: Backend>(path: impl AsRef<Path>, device: &B::Device) -> Result<OnnxModel<B>>;
pub fn read_net_from_onnx<B: Backend>(path: impl AsRef<Path>, device: &B::Device) -> Result<OnnxModel<B>>;
```

## Weight Loaders

```rust
pub struct WeightLoader;

impl WeightLoader {
    pub fn load_safetensors<B: Backend>(
        path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<HashMap<String, Tensor<B, 2>>>;

    pub fn load_bin<B: Backend>(
        path: impl AsRef<Path>,
        device: &B::Device,
        expected_shape: [usize; 2],
    ) -> Result<Tensor<B, 2>>;
}
```

## Helpers

```rust
pub fn blob_from_image<B: Backend>(
    image: &Image<B>,
    scalefactor: f64,
    size: Size<usize>,
    mean: Scalar,
    swap_rb: bool,
) -> Result<Tensor<B, 4>>;

pub fn nms_boxes(
    bboxes: &[Rect<usize>],
    scores: &[f32],
    score_threshold: f32,
    nms_threshold: f32,
) -> Vec<usize>;
```
