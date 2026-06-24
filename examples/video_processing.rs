use burn::backend::wgpu::Wgpu;
use observers::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Open a mock video capture
    println!("Opening video source...");
    let mut capture = VideoCapture::<Backend>::open("test_video.mp4", &device)?;

    // 2. Open a video writer target
    println!("Creating video destination...");
    let mut writer = VideoWriter::<Backend>::create("output_video.mp4", 640, 480, 30.0)?;

    // 3. Process first few frames
    println!("Processing frames...");
    let mut frame_count = 0;
    while let Some(frame) = capture.read()? {
        if frame_count >= 5 {
            break;
        }
        println!(" - Processed frame {} of size {:?}", frame_count, frame.shape());
        writer.write(&frame)?;
        frame_count += 1;
    }

    println!("Video processing example completed successfully.");
    Ok(())
}
