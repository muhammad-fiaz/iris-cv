// Demonstrates GIF creation and reading: generates animated frames,
// writes to GIF, reads back, and verifies frame data.

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

    // 1. Generate animated frames
    println!("Generating synthetic animation frames...");
    let width = 128;
    let height = 128;
    let num_frames = 30;

    let frames: Vec<Frame<Backend>> = (0..num_frames)
        .map(|i| {
            let mut flat_data = vec![0.0f32; 3 * height * width];
            let phase = (i as f32) / (num_frames as f32) * std::f32::consts::TAU;

            for y in 0..height {
                for x in 0..width {
                    let nx = x as f32 / width as f32;
                    let ny = y as f32 / height as f32;
                    let r = (nx * 2.0 + phase.sin() * 0.5).clamp(0.0, 1.0);
                    let g = (ny * 2.0 + phase.cos() * 0.5).clamp(0.0, 1.0);
                    let b = ((nx + ny) + (phase * 0.5).sin()).clamp(0.0, 1.0);
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
            let pts = Duration::from_millis(i as u64 * 33);
            Frame::new(img, pts, i)
                .with_duration(Duration::from_millis(33))
                .with_keyframe(i == 0)
        })
        .collect();

    println!("Generated {} frames ({}x{})", num_frames, width, height);

    // 2. Write to GIF
    println!("\nWriting GIF...");
    let opts = VideoWriteOptions {
        fps: 30.0,
        ..Default::default()
    };
    let mut writer = VideoFileWriter::<Backend>::create("output_gif.gif", width, height, &opts)?;

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

    // 3. Read the GIF back
    println!("\nReading GIF back...");
    let open_opts = VideoOpenOptions::default();
    let reader = VideoReader::<Backend>::open("output_gif.gif", &open_opts)?;

    println!(
        "Read {} frames: {}x{}, {:.1} fps",
        reader.frame_count(),
        reader.metadata().width,
        reader.metadata().height,
        reader.metadata().fps,
    );

    // 4. Verify roundtrip
    let read_frame = reader.get_frame(0)?;
    println!(
        "Frame 0: original shape={:?}, read shape={:?}",
        frames[0].shape(),
        read_frame.shape(),
    );

    // 5. Iterate all frames
    println!("\nIterating frames:");
    let iter = reader.iter();
    for frame in iter {
        if frame.index % 10 == 0 {
            println!(
                "  Frame {}: {}x{}, pts={:.3}s",
                frame.index,
                frame.width(),
                frame.height(),
                frame.pts.as_secs_f64(),
            );
        }
    }

    println!("\nGIF video example completed.");
    Ok(())
}
