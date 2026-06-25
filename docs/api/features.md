---
title: "Features Module Reference"
description: "API reference for Iris features module — ORB feature detection (FAST + BRIEF), template matching (6 methods), KeyPoint struct, BFMatcher, and FLANN matcher for descriptor matching."
keywords: ["features", "ORB", "FAST", "BRIEF", "keypoints", "feature detection", "descriptor matching", "BFMatcher", "template matching", "TM_CCOEFF_NORMED", "FLANN", "FlannMatcher"]
---

# Features Module Reference

Provides feature detection, descriptor computation, and descriptor matching.

## FeatureDetector

```rust
pub enum FeatureType {
    ORB,
    BRISK,
    AKAZE,
    SIFT,
}

pub struct FeatureDetector;

impl FeatureDetector {
    pub fn new(detector_type: FeatureType) -> Self;
    pub fn with_max_features(self, max: usize) -> Self;
    pub fn detect<B: Backend>(&self, image: &Image<B>) -> Result<Vec<KeyPoint>>;
    pub fn compute<B: Backend>(&self, image: &Image<B>, keypoints: &[KeyPoint]) -> Result<Tensor<B, 2>>;
}
```

## KeyPoint

```rust
pub struct KeyPoint {
    pub pt: Point<f64>,
    pub size: f64,
    pub angle: f64,
    pub response: f64,
    pub octave: i32,
    pub class_id: i32,
}

impl KeyPoint {
    pub fn new(x: f64, y: f64, size: f64) -> Self;
}
```

## BFMatcher

Brute-force matcher for comparing feature descriptors.

```rust
pub struct BFMatcher;

impl BFMatcher {
    pub fn new() -> Self;
    pub fn match_descriptors<B: Backend>(
        desc1: &Tensor<B, 2>,
        desc2: &Tensor<B, 2>,
    ) -> Result<Vec<DMatch>>;
}
```

### DMatch

```rust
pub struct DMatch {
    pub query_idx: usize,
    pub train_idx: usize,
    pub distance: f32,
}
```

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();
let img = Image::<Backend>::open("input.jpg", &device)?;

// Detect keypoints
let detector = FeatureDetector::new(FeatureType::ORB).with_max_features(500);
let keypoints = detector.detect(&img)?;
println!("Detected {} keypoints", keypoints.len());

// Compute ORB descriptors
let descriptors = detector.compute(&img, &keypoints)?;
println!("Descriptor shape: {:?}", descriptors.dims());
```

## Template Matching

```rust
pub enum TemplateMatchMethod {
    TmSqdiff,        // Sum of squared differences
    TmSqdiffNormed,  // Normalized SSD
    TmCcorr,         // Cross-correlation
    TmCcorrNormed,   // Normalized cross-correlation
    TmCcoeff,        // Cross-correlation coefficient
    TmCcoeffNormed,  // Normalized cross-correlation coefficient
}

pub fn template_match<B: Backend>(
    source: &Image<B>,
    template: &Image<B>,
    method: TemplateMatchMethod,
) -> Result<Tensor<B, 2>>;

// Also available as a method on Image:
impl<B: Backend> Image<B> {
    pub fn template_match(&self, template: &Image<B>, method: TemplateMatchMethod)
        -> Result<Tensor<B, 2>>;
}
```

### Example

```rust,ignore
use iris::prelude::*;

let device = WgpuDevice::default();
let source: Image<Wgpu> = Image::open("scene.png", &device)?;
let template: Image<Wgpu> = Image::open("object.png", &device)?;

// Find the object in the scene using normalized cross-correlation
let result = source.template_match(&template, TemplateMatchMethod::TmCcoeffNormed)?;

// Result is a 2D tensor where the maximum indicates best match position
let flat: Vec<f32> = result.clone().into_data().iter::<f32>().collect();
let max_idx = flat.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
let match_y = max_idx / result.dims()[1];
let match_x = max_idx % result.dims()[1];
println!("Best match at ({}, {})", match_x, match_y);
```

## FLANN Matcher

Fast Library for Approximate Nearest Neighbors matcher for efficient descriptor matching on large datasets.

```rust
pub struct FlannMatcher {
    k: usize,
    trees: usize,
    checks: usize,
}

impl FlannMatcher {
    pub fn new() -> Self;
    pub fn with_k(self, k: usize) -> Self;
    pub fn with_trees(self, trees: usize) -> Self;
    pub fn with_checks(self, checks: usize) -> Self;
    pub fn find_matches<B: Backend>(
        &self,
        desc1: &Tensor<B, 2>,
        desc2: &Tensor<B, 2>,
    ) -> Result<Vec<DMatch>>;
}
```

### Parameters

| Field | Default | Description |
|---|---|---|
| `k` | 2 | Number of nearest neighbors to find per query descriptor. |
| `trees` | 5 | Number of trees in the KD-tree index. More trees improve accuracy at cost of speed. |
| `checks` | 32 | Number of checks during search. Higher values improve accuracy. |

### Example

```rust,ignore
use iris::prelude::*;

let device = WgpuDevice::default();
let img1 = Image::<Wgpu>::open("frame1.jpg", &device)?;
let img2 = Image::<Wgpu>::open("frame2.jpg", &device)?;

let detector = FeatureDetector::new(FeatureType::ORB).with_max_features(1000);
let kp1 = detector.detect(&img1)?;
let kp2 = detector.detect(&img2)?;
let desc1 = detector.compute(&img1, &kp1)?;
let desc2 = detector.compute(&img2, &kp2)?;

// Fast approximate matching with FLANN
let matcher = FlannMatcher::new().with_k(2).with_trees(8);
let matches = matcher.find_matches(&desc1, &desc2)?;
println!("Found {} matches", matches.len());
```
