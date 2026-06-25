// Demonstrates video capture, writing, and frame iteration.
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

    // 1. Generate synthetic frames (gradient pattern)
    println!("Generating synthetic frames...");
    let mut frames: Vec<Frame<Backend>> = Vec::new();
    for i in 0..5 {
        let mut flat_data = vec![0.0f32; 3 * 480 * 640];
        for y in 0..480 {
            for x in 0..640 {
                let idx = y * 640 + x;
                flat_data[idx] = x as f32 / 640.0;
                flat_data[480 * 640 + idx] = y as f32 / 480.0;
                flat_data[2 * 480 * 640 + idx] = 1.0 - x as f32 / 640.0;
            }
        }
        let tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(flat_data, [3, 480, 640]),
            &device,
        );
        let img = Image::new(tensor);
        let frame = Frame::new(img, Duration::from_millis(i as u64 * 33), i)
            .with_duration(Duration::from_millis(33))
            .with_keyframe(i == 0);
        frames.push(frame);
    }

    // 2. Write frames to GIF
    println!("Writing video frames...");
    let opts = VideoWriteOptions {
        fps: 30.0,
        ..Default::default()
    };
    let mut writer =
        VideoFileWriter::<Backend>::create("output_video_processing.gif", 640, 480, &opts)?;

    for frame in &frames {
        writer.write_frame(frame)?;
    }
    writer.finish()?;

    println!(
        "Wrote {} frames to '{}' ({:.2}s)",
        writer.frame_count(),
        writer.output_path().display(),
        writer.duration().as_secs_f64(),
    );

    // 3. FrameIterator demo
    println!("\nFrameIterator demo:");
    let mut iter = FrameIterator::new(frames).with_loop();
    println!("Total frames: {}", iter.total_frames());

    for _ in 0..15 {
        let frame = iter.next().unwrap();
        if frame.index == 0 && iter.current_index() == 1 {
            print!("  Loop at frame 0, ");
        }
    }
    println!("iterated 15 frames successfully.");

    println!("\nVideo processing example completed.");
    Ok(())
}
