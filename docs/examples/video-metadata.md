---
title: "Video Metadata Example"
description: "Format detection, stream info, pixel formats with Iris."
keywords: ["video", "metadata", "format detection"]
---

# Video Metadata

Demonstrates video metadata: format detection, stream info, pixel formats, and synthetic metadata construction.

```bash
cargo run --example video_metadata --features wgpu
```

## Source

```rust
// Demonstrates video metadata: format detection, stream info, pixel formats,
// and synthetic metadata construction. No image files required.

use burn::backend::wgpu::Wgpu;
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. VideoMetadata for different container formats
    println!("=== VideoMetadata Examples ===\n");

    let formats = [
        ContainerFormat::Gif,
        ContainerFormat::Png,
        ContainerFormat::Mp4,
        ContainerFormat::Mkv,
        ContainerFormat::WebP,
        ContainerFormat::Qoi,
    ];

    for format in &formats {
        let meta = VideoMetadata::synthetic(1920, 1080, 30.0, 300);
        println!(
            "{:?}: {}x{}, {:.1} fps, {:.1}s, aspect={:.2}",
            format,
            meta.width,
            meta.height,
            meta.fps,
            meta.duration.as_secs_f64(),
            meta.aspect_ratio(),
        );
    }

    // 2. ContainerFormat detection from file extensions
    println!("\n=== ContainerFormat Detection ===\n");

    let test_files = [
        "animation.gif",
        "photo.png",
        "video.mp4",
        "movie.mkv",
        "image.webp",
        "texture.qoi",
        "photo.jpg",
        "output.avi",
        "stream.webm",
        "unknown.xyz",
    ];

    for file in &test_files {
        let format = ContainerFormat::from_path(file);
        println!("  {} -> {:?} (ext: {})", file, format, format.extension());
    }

    // 3. PixelFormat channels
    println!("\n=== PixelFormat Channels ===\n");

    let pixel_formats = [
        PixelFormat::Gray8,
        PixelFormat::Rgb8,
        PixelFormat::Rgba8,
        PixelFormat::Rgb16,
        PixelFormat::Rgba16,
        PixelFormat::Rgb32F,
        PixelFormat::Rgba32F,
    ];

    for pf in &pixel_formats {
        println!("  {:?}: {} channels", pf, pf.channels());
    }

    // 4. Metadata with streams
    println!("\n=== Metadata with Streams ===\n");

    let mut meta = VideoMetadata::synthetic(1920, 1080, 30.0, 300);
    meta.streams = vec![
        StreamInfo {
            index: 0,
            stream_type: StreamType::Video,
            codec: "h264".to_string(),
            width: 1920,
            height: 1080,
            fps: 30.0,
            duration: std::time::Duration::from_secs(10),
            frame_count: 300,
            pixel_format: PixelFormat::Rgb8,
            rotation: 0,
            bit_rate: 5_000_000,
        },
        StreamInfo {
            index: 1,
            stream_type: StreamType::Audio,
            codec: "aac".to_string(),
            width: 0,
            height: 0,
            fps: 0.0,
            duration: std::time::Duration::from_secs(10),
            frame_count: 0,
            pixel_format: PixelFormat::Gray8,
            rotation: 0,
            bit_rate: 128_000,
        },
    ];
    meta.has_audio = true;

    println!("Video file metadata:");
    println!("  Format: {:?}", meta.format);
    println!("  Size: {}x{}", meta.width, meta.height);
    println!("  Duration: {:.1}s", meta.duration.as_secs_f64());
    println!("  FPS: {:.1}", meta.fps);
    println!("  Aspect ratio: {:.2}", meta.aspect_ratio());
    println!("  Video streams: {}", meta.video_stream_count());
    println!("  Audio streams: {}", meta.audio_stream_count());

    for stream in &meta.streams {
        println!(
            "  Stream {}: {:?} - {} ({}, {}x{})",
            stream.index,
            stream.stream_type,
            stream.codec,
            stream.bit_rate,
            stream.width,
            stream.height,
        );
    }

    println!("\nVideo metadata example completed.");
    Ok(())
}
```
