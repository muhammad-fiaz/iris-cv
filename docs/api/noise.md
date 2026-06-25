---
title: "Noise Module Reference"
description: "API reference for Iris noise module — add_gaussian_noise, add_salt_pepper_noise, and add_speckle_noise for noise simulation."
keywords: ["noise", "gaussian noise", "salt and pepper", "speckle noise", "image noise", "noise generation"]
---

# Noise Module Reference

Provides noise generation functions for simulating various types of image degradation.

## Operations

```rust
impl<B: Backend> Image<B> {
    pub fn add_gaussian_noise(&self, mean: f32, std_dev: f32) -> Result<Self>;
    pub fn add_salt_pepper_noise(&self, amount: f32) -> Result<Self>;
    pub fn add_speckle_noise(&self, std_dev: f32) -> Result<Self>;
}
```

## Noise Types

### Gaussian Noise

Adds additive Gaussian noise: `output = pixel + N(mean, std_dev)`. Uses the Marsaglia polar method for Gaussian random number generation.

| Parameter | Description |
|---|---|
| `mean` | Mean of the Gaussian distribution (typically 0.0). |
| `std_dev` | Standard deviation of the noise (e.g., 0.05 for mild noise). |

### Salt-and-Pepper Noise

Adds impulse noise by randomly setting pixels to 0.0 (pepper) or 1.0 (salt).

| Parameter | Description |
|---|---|
| `amount` | Fraction of pixels to corrupt, in range `[0.0, 1.0]`. |

### Speckle Noise

Adds multiplicative noise: `output = pixel + pixel * N(0, std_dev)`. Common in radar and medical imaging.

| Parameter | Description |
|---|---|
| `std_dev` | Standard deviation of the multiplicative noise (e.g., 0.1). |

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();
let img = Image::<Backend>::open("input.jpg", &device)?;

// Add mild Gaussian noise
let noisy = img.add_gaussian_noise(0.0, 0.05)?;

// Add 10% salt-and-pepper noise
let peppered = img.add_salt_pepper_noise(0.1)?;

// Add speckle noise
let speckled = img.add_speckle_noise(0.1)?;

noisy.save("gaussian_noisy.png")?;
peppered.save("salt_pepper_noisy.png")?;
speckled.save("speckle_noisy.png")?;
```
