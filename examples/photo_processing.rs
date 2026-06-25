// Demonstrates photo processing: non-local means denoising and Mertens exposure fusion.
// Loads a real image and applies denoising and exposure merging.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load a real image as the base frame
    let base_img: Image<Backend> = Image::open("assets/images/gradient.png", &device)?;

    // 1. Non-Local Means Denoising
    println!("Applying fast Non-Local Means Denoising...");
    let denoised = Photo::fast_nl_means_denoising(&base_img, 10.0, 3, 5)?;
    denoised.save("output_photo_denoised.png")?;

    // 2. Mertens Exposure Fusion
    println!("Performing Mertens Exposure Fusion...");
    let exposure1 = base_img.clone();
    let exposure2 = Image::new(base_img.tensor.clone().add_scalar(0.2));
    let exposure3 = Image::new(base_img.tensor.clone().sub_scalar(0.2));

    let fusion = MergeMertens::default();
    let merged = fusion.process(&[exposure1, exposure2, exposure3])?;
    merged.save("output_photo_fusion.png")?;

    println!("Photo processing completed. Outputs saved.");
    Ok(())
}
