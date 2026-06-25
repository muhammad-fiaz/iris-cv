---
title: "Video Module Reference"
description: "API reference for Iris video module — VideoCapture, VideoReader, VideoWriter, Frame, FrameIterator, VideoMetadata, and video utilities."
keywords: ["video", "VideoCapture", "VideoReader", "VideoWriter", "Frame", "FrameIterator", "metadata", "video processing"]
---

# Video Module Reference

Provides video capture, reading, writing, frame iteration, and metadata extraction.

## VideoCapture (Legacy)

```rust
pub struct VideoCapture<B: Backend>;

impl<B: Backend> VideoCapture<B> {
    pub fn open(path: impl AsRef<Path>, device: &B::Device) -> Result<Self>;
    pub fn read(&mut self) -> Result<Option<Image<B>>>;
}
```

## VideoReader

Real video file reader for reading frames from video files.

```rust
pub struct VideoReader;

impl VideoReader {
    pub fn open(path: impl AsRef<Path>, options: VideoOpenOptions) -> Result<Self>;
    pub fn read_frame<B: Backend>(&mut self, device: &B::Device) -> Result<Option<Frame<B>>>;
    pub fn seek(&mut self, frame_number: u64, mode: SeekMode) -> Result<()>;
    pub fn metadata(&self) -> VideoMetadata;
}
```

### VideoOpenOptions

```rust
pub struct VideoOpenOptions {
    pub stream_type: StreamType,
    pub pixel_format: PixelFormat,
}
```

### SeekMode

```rust
pub enum SeekMode {
    Fast,
    Accurate,
}
```

## VideoWriter

```rust
pub struct VideoWriter;

impl VideoWriter {
    pub fn create(
        path: impl AsRef<Path>,
        options: VideoWriteOptions,
    ) -> Result<Self>;
    pub fn write<B: Backend>(&mut self, frame: &Frame<B>) -> Result<()>;
    pub fn finalize(&mut self) -> Result<()>;
}
```

### VideoWriteOptions

```rust
pub struct VideoWriteOptions {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub output_format: OutputFormat,
}
```

### OutputFormat

```rust
pub enum OutputFormat {
    MP4,
    AVI,
    WebM,
}
```

## Frame

```rust
pub struct Frame<B: Backend> {
    pub image: Image<B>,
    pub timestamp: std::time::Duration,
    pub frame_number: u64,
}

impl<B: Backend> Frame<B> {
    pub fn new(image: Image<B>, timestamp: std::time::Duration, frame_number: u64) -> Self;
    pub fn width(&self) -> usize;
    pub fn height(&self) -> usize;
}
```

## FrameIterator

```rust
pub struct FrameIterator<B: Backend>;

impl<B: Backend> FrameIterator<B> {
    pub fn new(frames: Vec<Frame<B>>) -> Self;
    pub fn total_frames(&self) -> usize;
}
```

## VideoMetadata

```rust
pub struct VideoMetadata {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub frame_count: u64,
    pub duration_secs: f64,
    pub codec: String,
}

impl VideoMetadata {
    pub fn synthetic(width: usize, height: usize, fps: f64, frame_count: u64) -> Self;
}
```

## Utility Functions

```rust
pub fn load_animated_image<B: Backend>(path: impl AsRef<Path>, device: &B::Device) -> Result<Vec<Frame<B>>>;
pub fn load_image_sequence<B: Backend>(pattern: &str, device: &B::Device) -> Result<Vec<Frame<B>>>;
```

## Example

```rust
use iris::prelude::*;
use burn::backend::wgpu::Wgpu;

type Backend = Wgpu;
let device = Default::default();

// Legacy capture
let mut capture = VideoCapture::<Backend>::open("video.mp4", &device)?;
while let Some(frame) = capture.read()? {
    println!("Frame shape: {:?}", frame.shape());
}

// Write video
let mut writer = LegacyVideoWriter::<Backend>::create("output.mp4", 640, 480, 30.0)?;
writer.write(&frame)?;
```
