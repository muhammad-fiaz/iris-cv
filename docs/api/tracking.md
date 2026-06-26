---
title: "Tracking Module Reference"
description: "API reference for Iris tracking module — MOSSE tracker, KCF/CSRT tracker types, MeanShift tracker, and background subtraction."
keywords: ["tracking", "MOSSE", "KCF", "CSRT", "tracker", "background subtraction", "object tracking", "MeanShift", "MeanShiftTracker"]
---

# Tracking Module Reference

Provides object tracking and background subtraction algorithms.

## Tracker

```rust
pub enum TrackerType {
    KCF,
    CSRT,
    MOSSE,
}

pub struct Tracker<B: Backend> {
    pub tracker_type: TrackerType,
    pub bbox: Option<Rect<usize>>,
}

impl<B: Backend> Tracker<B> {
    pub fn new(tracker_type: TrackerType) -> Self;
    pub fn init(&mut self, image: &Image<B>, bbox: Rect<usize>) -> Result<()>;
    pub fn update(&mut self, image: &Image<B>) -> Result<Rect<usize>>;
}
```

## BackgroundSubtractor

```rust
pub struct BackgroundSubtractor<B: Backend> {
    learning_rate: f32,
    threshold: f32,
    background: Option<Image<B>>,
}

impl<B: Backend> BackgroundSubtractor<B> {
    pub fn new(learning_rate: f32, threshold: f32) -> Self;
    pub fn apply(&mut self, frame: &Image<B>) -> Result<Image<B>>;
}
```

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();

// Background subtraction
let mut subtractor = BackgroundSubtractor::new(0.1, 0.05);
let mask = subtractor.apply(&frame)?;

// Object tracking
let mut tracker = Tracker::new(TrackerType::MOSSE);
let init_bbox = Rect::new(50, 50, 100, 100);
tracker.init(&frame1, init_bbox)?;

// Track across frames
let bbox = tracker.update(&frame2)?;
println!("Object at: {:?}", bbox);
```

## MeanShift Tracker

Iterative kernel-based tracker that locates the object by shifting the search window toward the peak of the color histogram probability distribution.

```rust
pub struct MeanShiftTracker {
    // internal state
}

impl MeanShiftTracker {
    pub fn new() -> Self;
    pub fn init<B: Backend>(&mut self, image: &Image<B>, roi: Rect<usize>) -> Result<()>;
    pub fn update<B: Backend>(&mut self, image: &Image<B>) -> Result<Rect<usize>>;
}
```

### Methods

| Method | Description |
|---|---|
| `new()` | Creates a new MeanShift tracker with default parameters. |
| `init(image, roi)` | Initializes the tracker with the target region of interest (histogram model). |
| `update(image)` | Performs one MeanShift iteration and returns the updated bounding box. |

### Example

```rust
use iris::prelude::*;

let device = WgpuDevice::default();
let frame1 = Image::<Wgpu>::open("frame1.jpg", &device)?;
let frame2 = Image::<Wgpu>::open("frame2.jpg", &device)?;

let mut tracker = MeanShiftTracker::new();
let roi = Rect::new(120, 80, 64, 64);
tracker.init(&frame1, roi)?;

let bbox = tracker.update(&frame2)?;
println!("Tracked object at: {:?}", bbox);
```
