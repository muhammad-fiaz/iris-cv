// Demonstrates loading a sequence of numbered PNG images into Frame objects,
// creating a FrameIterator, and seeking to specific frames.

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

    // 1. Generate numbered PNG files on disk
    println!("Creating image sequence on disk...");
    let seq_dir = std::env::temp_dir().join("iris_image_sequence");
    std::fs::create_dir_all(&seq_dir)?;

    let width = 64;
    let height = 64;
    let num_frames = 10;

    for i in 0..num_frames {
        let mut flat_data = vec![0.0f32; 3 * height * width];
        let hue = i as f32 / num_frames as f32;

        for y in 0..height {
            for x in 0..width {
                let nx = x as f32 / width as f32;
                let ny = y as f32 / height as f32;
                flat_data[y * width + x] = (nx + hue).clamp(0.0, 1.0);
                flat_data[height * width + y * width + x] = (ny + hue * 0.5).clamp(0.0, 1.0);
                flat_data[2 * height * width + y * width + x] = ((nx + ny) * 0.5).clamp(0.0, 1.0);
            }
        }

        let tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(flat_data, [3, height, width]),
            &device,
        );
        let img = Image::new(tensor);
        let path = seq_dir.join(format!("frame_{:04}.png", i));
        img.save(&path)?;
    }

    println!("Saved {} PNG frames to {}", num_frames, seq_dir.display());

    // 2. Load the sequence
    println!("\nLoading image sequence...");
    let loaded_frames =
        load_image_sequence::<Backend>(&seq_dir, "frame", &device, 30.0)?;

    println!("Loaded {} frames", loaded_frames.len());

    // 3. Verify frame properties
    println!("\nFrame details:");
    for frame in &loaded_frames {
        println!(
            "  Frame {}: {}x{}, pts={:.3}s, duration={:.3}s, keyframe={}",
            frame.index,
            frame.width(),
            frame.height(),
            frame.pts.as_secs_f64(),
            frame.duration.as_secs_f64(),
            frame.is_keyframe,
        );
    }

    // 4. Process with FrameIterator
    println!("\nProcessing with FrameIterator:");
    let mut iter = FrameIterator::new(loaded_frames);
    let mut processed_count = 0;

    for frame in iter.by_ref() {
        let data: Vec<f32> = frame.image.tensor.to_data().into_vec().unwrap();
        let avg: f32 = data.iter().sum::<f32>() / data.len() as f32;

        if frame.index % 3 == 0 {
            println!("  Frame {}: avg brightness = {:.4}", frame.index, avg);
        }
        processed_count += 1;
    }

    println!("Processed {} frames total", processed_count);

    // 5. Seek demo
    println!("\nSeek demo:");
    let mut seek_iter = FrameIterator::new(
        (0..20)
            .map(|i| {
                let data = TensorData::new(vec![0.5f32; 3 * 32 * 32], [3, 32, 32]);
                let tensor = Tensor::<Backend, 3>::from_data(data, &device);
                Frame::new(Image::new(tensor), Duration::from_millis(i as u64 * 50), i)
            })
            .collect(),
    );

    seek_iter.seek(5)?;
    println!("  Seeked to index {}", seek_iter.current_index());
    let frame = seek_iter.next().unwrap();
    println!("  Got frame {}", frame.index);

    // Cleanup
    std::fs::remove_dir_all(&seq_dir)?;

    println!("\nImage sequence example completed.");
    Ok(())
}
