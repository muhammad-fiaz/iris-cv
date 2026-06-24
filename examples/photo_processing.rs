use burn::backend::wgpu::Wgpu;
use observers::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate noisy image
    let w = 128;
    let h = 128;
    let flat_noisy = vec![0.5f32; 3 * h * w];
    let noisy_img = Image::new(Tensor::<Backend, 3>::from_data(TensorData::new(flat_noisy, [3, h, w]), &device));

    // 2. Perform Non-Local Means Denoising
    println!("Applying fast Non-Local Means Denoising...");
    let denoised = Photo::fast_nl_means_denoising(&noisy_img, 10.0)?;
    denoised.save("output_denoised.png")?;

    // 3. Perform Mertens Exposure Fusion
    println!("Performing Mertens Exposure Fusion on multiple exposures...");
    let exposure1 = noisy_img.clone();
    let exposure2 = Image::new(noisy_img.tensor.clone().add_scalar(0.2));
    let exposure3 = Image::new(noisy_img.tensor.clone().sub_scalar(0.2));

    let fusion = MergeMertens::default();
    let merged = fusion.process(&[exposure1, exposure2, exposure3])?;
    merged.save("output_mertens_fusion.png")?;

    println!("Photo processing example completed successfully.");
    Ok(())
}
