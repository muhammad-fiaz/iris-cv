---
title: "Video Module Reference"
description: "API reference for Iris video module — VideoCapture, VideoReader, VideoWriter, Frame, FrameIterator, VideoMetadata, and video utilities."
keywords: ["video", "VideoCapture", "VideoReader", "VideoWriter", "Frame", "FrameIterator", "metadata", "video processing"]
canonical: "https://muhammad-fiaz.github.io/iris-cv/api/video"
---

# Video Module Reference

Provides video capture, reading, writing, frame iteration, and metadata extraction.

::: note
This module is under active development. API signatures may change between versions.
:::

## VideoCapture (Legacy)

```rust
pub struct VideoCapture<B: Backend>;

impl<B: Backend> VideoCapture<B> {
    pub fn open(path: impl AsRef<Path>, device: &B::Device) -> Result<Self>;
    pub fn read(&mut self) -> Result<Option<Image<B>>>;
}
```

## VideoReader

High-level video reader that loads all frames into memory for random access.

```rust
pub struct VideoReader<B: Backend> { /* ... */ }

impl<B: Backend> VideoReader<B> {
    pub fn open(path: impl AsRef<Path>, options: &VideoOpenOptions) -> Result<Self>;
    pub fn frames(&self) -> &[Frame<B>];
    pub fn frame(&self, index: usize) -> Option<&Frame<B>>;
    pub fn metadata(&self) -> &VideoMetadata;
    pub fn frame_count(&self) -> usize;
    pub fn duration(&self) -> Duration;
}
```

### VideoOpenOptions

```rust
pub struct VideoOpenOptions {
    pub seek_mode: SeekMode,
    pub max_frames: usize,
    pub preload_all: bool,
    pub target_width: usize,
    pub target_height: usize,
    pub sequence_fps: f64,
    pub sequence_pattern: String,
}
```

### SeekMode

```rust
pub enum SeekMode {
    ByKeyframe,
    Exact,
    ByTimestamp,
}
```

## VideoWriter

```rust
pub struct VideoWriter<B: Backend> { /* ... */ }

impl<B: Backend> VideoWriter<B> {
    pub fn create(
        output_path: impl AsRef<Path>,
        width: usize,
        height: usize,
        options: &VideoWriteOptions,
    ) -> Result<Self>;
    pub fn write_frame(&mut self, frame: &Frame<B>) -> Result<()>;
    pub fn frame_count(&self) -> usize;
    pub fn duration(&self) -> Duration;
    pub fn finish(&mut self) -> Result<()>;
}
```

### VideoWriteOptions

```rust
pub struct VideoWriteOptions {
    pub format: OutputFormat,
    pub fps: f64,
    pub gif_loops: u32,
    pub jpeg_quality: u8,
}
```

### OutputFormat

```rust
pub enum OutputFormat {
    Gif,
    PngSequence,
    JpegSequence,
    QoiSequence,
}
```

## Frame

```rust
pub struct Frame<B: Backend> {
    pub image: Image<B>,
    pub pts: Duration,
    pub duration: Duration,
    pub index: usize,
    pub is_keyframe: bool,
}

impl<B: Backend> Frame<B> {
    pub fn new(image: Image<B>, pts: Duration, index: usize) -> Self;
    pub fn keyframe(image: Image<B>, pts: Duration, index: usize) -> Self;
    pub fn with_duration(self, duration: Duration) -> Self;
    pub fn width(&self) -> usize;
    pub fn height(&self) -> usize;
}
```

## FrameIterator

```rust
pub struct FrameIterator<B: Backend> { /* ... */ }

impl<B: Backend> FrameIterator<B> {
    pub fn new(frames: Vec<Frame<B>>) -> Self;
    pub fn with_loop(self) -> Self;
    pub fn total_frames(&self) -> usize;
    pub fn current_index(&self) -> usize;
    pub fn seek(&mut self, index: usize) -> Result<()>;
    pub fn seek_to_time(&mut self, time: Duration, fps: f64) -> Result<()>;
}
```

## VideoMetadata

```rust
pub struct VideoMetadata {
    pub format: ContainerFormat,
    pub duration: Duration,
    pub fps: f64,
    pub width: usize,
    pub height: usize,
    pub frame_count: usize,
    pub file_size: u64,
    pub codec: String,
}

impl VideoMetadata {
    pub fn synthetic(width: usize, height: usize, fps: f64, frame_count: usize) -> Self;
}
```

## Utility Functions

```rust
pub fn load_animated_image<B: Backend>(path: impl AsRef<Path>, device: &B::Device) -> Result<Vec<Frame<B>>>;
pub fn load_image_sequence<B: Backend>(pattern: &str, device: &B::Device) -> Result<Vec<Frame<B>>>;
```

## Legacy

```rust
pub struct LegacyVideoWriter<B: Backend>;

impl<B: Backend> LegacyVideoWriter<B> {
    pub fn create(path: impl AsRef<Path>, width: usize, height: usize, fps: f64) -> Result<Self>;
    pub fn write(&mut self, frame: &Image<B>) -> Result<()>;
}
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

// Read real video
let options = VideoOpenOptions::default();
let reader = VideoReader::open("video.gif", &options)?;
println!("Frames: {}, FPS: {}", reader.frame_count(), reader.metadata().fps);

// Write video
let write_options = VideoWriteOptions {
    format: OutputFormat::Gif,
    fps: 15.0,
    ..Default::default()
};
let mut writer = VideoWriter::<Backend>::create("output.gif", 640, 480, &write_options)?;
writer.write_frame(&frame)?;
writer.finish()?;
```
