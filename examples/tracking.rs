use burn::backend::wgpu::Wgpu;
use observers::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate test frames
    let w = 100;
    let h = 100;
    let flat1 = vec![0.5f32; 3 * h * w];
    // Slightly different second frame to trigger motion detection
    let mut flat2 = vec![0.5f32; 3 * h * w];
    for y in 40..60 {
        for x in 40..60 {
            flat2[y * w + x] = 0.9;
        }
    }

    let img1 = Image::new(Tensor::<Backend, 3>::from_data(TensorData::new(flat1, [3, h, w]), &device));
    let img2 = Image::new(Tensor::<Backend, 3>::from_data(TensorData::new(flat2, [3, h, w]), &device));

    // 2. Background subtraction
    println!("Applying BackgroundSubtractor...");
    let mut bs = BackgroundSubtractor::new(0.1, 0.05);
    let _mask1 = bs.apply(&img1)?;
    let mask2 = bs.apply(&img2)?;

    println!("Detected foreground mask shape: {:?}", mask2.shape());
    mask2.save("output_foreground_mask.png")?;

    // 3. Object Tracking
    println!("Initializing object tracker...");
    let mut tracker = Tracker::new(TrackerType::CSRT);
    let init_bbox = Rect::new(40, 40, 20, 20);
    tracker.init(&img1, init_bbox)?;

    println!("Updating tracker with next frame...");
    let updated_bbox = tracker.update(&img2)?;
    println!("Updated object bounding box: {:?}", updated_bbox);

    println!("Tracking example completed successfully.");
    Ok(())
}
