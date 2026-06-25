---
title: "Image Representation"
description: "How Iris represents images as Burn tensors — create, load, save, crop, flip, rotate, and query shape properties of multi-channel images."
keywords: ["image representation", "tensor", "channels", "height", "width", "image loading", "image I/O"]
---

# Image Representation

In Iris, images are represented by the `Image<B>` struct, where `B` is a Burn `Backend` type.

An `Image` wraps a 3D Burn tensor with the shape format `[Channels, Height, Width]` containing float values in the range `[0.0, 1.0]`.

## Creating Images

You can create an `Image` from a raw Burn `Tensor` or by loading an image file.

```rust
use burn::backend::wgpu::Wgpu;
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    // Create a mock image tensor [3 channels, 480 height, 640 width]
    let flat_data = vec![0.5f32; 3 * 480 * 640];
    let tensor = Tensor::<Backend, 3>::from_data(
        TensorData::new(flat_data, [3, 480, 640]),
        &device
    );
    let image = Image::new(tensor);

    println!("Created image with dimensions: {:?}", image.shape());
    Ok(())
}
```

## Loading and Saving

Iris uses the `image` crate internally to handle file decoders.

```rust
type Backend = Wgpu;
let device = Default::default();

// Load an image from file
let img = Image::<Backend>::open("input.png", &device)?;

// Save an image to file
img.save("output.png")?;
```

## Shape Properties

Convenient properties are exposed on the `Image` struct:

| Method | Description |
|---|---|
| `width()` | Returns the image width (dimension 2 of the tensor). |
| `height()` | Returns the image height (dimension 1 of the tensor). |
| `channels()` | Returns the number of channels (dimension 0 of the tensor). |
| `shape()` | Returns the `[C, H, W]` shape array. |

## Color Conversions

```rust
// Convert to grayscale
let gray = image.grayscale()?;

// Convert grayscale back to RGB
let rgb = gray.to_rgb()?;

// Convert between color spaces
let hsv = image.rgb_to_hsv()?;
let rgb_back = hsv.hsv_to_rgb()?;
```

## Geometric Transforms

```rust
// Resize to new dimensions
let resized = image.resize(640, 480)?;

// Rotate by angle in degrees
let rotated = image.rotate(90)?;

// Flip horizontally (axis = 1)
let flipped = image.flip(1)?;

// Crop a region
let cropped = image.crop(Rect::new(10, 10, 200, 200))?;
```

## Gaussian Pyramid

```rust
// Downsample to create a pyramid level
let pyramid = image.gaussian_pyramid()?;
```
