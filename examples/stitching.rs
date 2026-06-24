use burn::backend::wgpu::Wgpu;
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate two mock overlapping images
    let w = 120;
    let h = 80;
    let flat_left = vec![0.3f32; 3 * h * w];
    let flat_right = vec![0.4f32; 3 * h * w];

    let left = Image::new(Tensor::<Backend, 3>::from_data(
        TensorData::new(flat_left, [3, h, w]),
        &device,
    ));
    let right = Image::new(Tensor::<Backend, 3>::from_data(
        TensorData::new(flat_right, [3, h, w]),
        &device,
    ));

    // 2. Perform panorama stitching
    println!("Stitching images together...");
    let stitcher = Stitcher;
    let panorama = stitcher.stitch(&[left, right])?;

    println!("Stitched panorama shape: {:?}", panorama.shape());
    panorama.save("output_panorama.png")?;

    println!("Stitching example completed successfully.");
    Ok(())
}
