---
title: "Deep Neural Networks (DNN)"
description: "Load and run ONNX, Safetensors, and PyTorch bin models with Iris's native DNN module powered by the Burn framework."
keywords: ["DNN", "deep learning", "ONNX", "safetensors", "neural network", "inference", "model loading"]
---

# Deep Neural Networks (DNN)

Deep learning modules in Iris are built natively on the **Burn** framework. This allows you to construct model structures, load weights, and execute forward passes across multiple backends (WGPU, CUDA, LibTorch, Ndarray).

## Loading Pretrained Model Weights

### Safetensors format

Loads weights from standard `.safetensors` files using the `safetensors` crate.

```rust
let device = Default::default();
let weights = WeightLoader::load_safetensors::<Backend>("weights/face_detector.safetensors", &device)?;
```

### PyTorch Bin format

Loads weights from raw flat binary streams.

```rust
let weight_tensor = WeightLoader::load_bin::<Backend>("weights/model.bin", &device, [100, 100])?;
```

## Running ONNX Models

Iris supports importing ONNX model pipelines.

```rust
// Load an ONNX model from file
let model = OnnxModel::load("weights/object_detector.onnx", &device)?;

// Preprocess the input image to get a [1, C, H, W] tensor
let input_tensor = model.preprocess(&image)?;

// Evaluate the neural network
let output_tensor: Tensor<Backend, 4> = model.predict_raw(input_tensor)?;
```

### Direct Builders

```rust
let model = read_net::<Backend>("weights/model.onnx", &device)?;
let model = read_net_from_onnx::<Backend>("weights/model.onnx", &device)?;
```

## Input Preprocessing (Blobs)

The `blob_from_image` helper handles resizing, scaling, and mean subtraction to prepare images for neural network entry.

```rust
let size = Size::new(224, 224);
let mean = Scalar::new(0.485, 0.456, 0.406, 0.0);
let scale = 1.0 / 255.0;

let blob = blob_from_image(&image, scale, size, mean, true)?; // [1, C, 224, 224]
```

## Non-Maximum Suppression (NMS)

Filters overlapping bounding boxes based on class confidence scores and Intersection over Union (IoU) values.

```rust
let kept_indices = nms_boxes(&bboxes, &scores, 0.5, 0.4);
```
