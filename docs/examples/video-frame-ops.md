---
title: "Video Frame Operations Example"
description: "Frame differencing, motion magnitudes, batch tensor, seeking, and looping with Iris."
keywords: ["video", "frame operations", "motion", "batch"]
---

# Video Frame Operations

Demonstrates frame differencing, motion magnitudes, batch tensor conversion, random access, seeking, and looping.

```bash
cargo run --example video_frame_ops --features wgpu
```

## Source

```rust
// Demonstrates video frame operations: frame differencing, motion magnitudes,
// batch tensor conversion, random access, seeking, and looping.
// Uses synthetic gradient frames (no external video file required).

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;
use std::time::Duration;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate synthetic frames with motion
    println!("Generating synthetic frames with motion...");
    let width = 64;
    let height = 64;
    let num_frames = 10;

    let frames: Vec<Frame<Backend>> = (0..num_frames)
        .map(|i| {
            let mut flat_data = vec![0.0f32; 3 * height * width];
            let offset = (i as f32) / (num_frames as f32);

            for y in 0..height {
                for x in 0..width {
                    let nx = x as f32 / width as f32;
                    let ny = y as f32 / height as f32;
                    let r = ((nx + offset) % 1.0).clamp(0.0, 1.0);
                    let g = ((ny + offset * 0.5) % 1.0).clamp(0.0, 1.0);
                    let b = (((nx + ny) * 0.5 + offset * 0.3) % 1.0).clamp(0.0, 1.0);
                    flat_data[y * width + x] = r;
                    flat_data[height * width + y * width + x] = g;
                    flat_data[2 * height * width + y * width + x] = b;
                }
            }

            let tensor = Tensor::<Backend, 3>::from_data(
                TensorData::new(flat_data, [3, height, width]),
                &device,
            );
            let img = Image::new(tensor);
            Frame::new(img, Duration::from_millis(i as u64 * 33), i)
                .with_duration(Duration::from_millis(33))
                .with_keyframe(i == 0 || i % 5 == 0)
        })
        .collect();

    // 2. Build in-memory VideoReader
    println!("Building in-memory VideoReader...");
    let meta = VideoMetadata::synthetic(width, height, 30.0, num_frames);
    let reader = VideoReader::from_frames(frames.clone(), meta);

    println!(
        "Reader: {} frames, {}x{}, {:.1} fps",
        reader.frame_count(),
        reader.metadata().width,
        reader.metadata().height,
        reader.metadata().fps,
    );

    // 3. Frame differencing
    println!("\n=== Frame Differencing ===");
    let diffs = reader.frame_differences()?;
    println!("Computed {} frame differences", diffs.len());

    for (i, diff) in diffs.iter().enumerate() {
        let data: Vec<f32> = diff.to_data().into_vec().unwrap();
        let max_val: f32 = data.iter().cloned().fold(0.0f32, f32::max);
        let mean_val: f32 = data.iter().sum::<f32>() / data.len() as f32;
        println!("  Diff {}: max={:.4}, mean={:.6}", i, max_val, mean_val);
    }

    // 4. Motion magnitudes
    println!("\n=== Motion Magnitudes ===");
    let magnitudes = reader.motion_magnitudes()?;
    for (i, mag) in magnitudes.iter().enumerate() {
        println!("  Motion {}: {:.4}", i, mag);
    }

    // 5. Batch tensor
    println!("\n=== Batch Tensor ===");
    let batch = reader.to_batch_tensor()?;
    println!("Batch tensor shape: {:?}", batch.dims());

    // 6. Random access
    println!("\n=== Random Access ===");
    let frame = reader.get_frame(5)?;
    println!(
        "Frame 5: index={}, pts={:.3}s, keyframe={}",
        frame.index,
        frame.pts.as_secs_f64(),
        frame.is_keyframe,
    );

    let range = reader.get_range(2, 6)?;
    println!("Frames 2..6: {} frames", range.len());

    // 7. Seek to time
    println!("\n=== Seek to Time ===");
    let frame = reader.seek_to_time(Duration::from_millis(150))?;
    println!(
        "Seeked to 150ms: frame {}, pts={:.3}s",
        frame.index,
        frame.pts.as_secs_f64(),
    );

    // 8. Looping iterator
    println!("\n=== Looping Iterator ===");
    let mut loop_iter = reader.loop_iter();
    let mut loop_count = 0;
    for _ in 0..25 {
        let frame = loop_iter.next().unwrap();
        if frame.index == 0 {
            loop_count += 1;
        }
    }
    println!("Looped 25 frames, hit start {} times", loop_count);

    // 9. Write to GIF
    println!("\n=== Write to GIF ===");
    let opts = VideoWriteOptions {
        fps: 30.0,
        ..Default::default()
    };
    let mut writer =
        VideoFileWriter::<Backend>::create("output_video_frame_ops.gif", width, height, &opts)?;

    for frame in &frames {
        writer.write_frame(frame)?;
    }
    writer.finish()?;

    println!(
        "Saved '{}' ({} frames, {:.2}s)",
        writer.output_path().display(),
        writer.frame_count(),
        writer.duration().as_secs_f64(),
    );

    println!("\nVideo frame operations example completed.");
    Ok(())
}
```
