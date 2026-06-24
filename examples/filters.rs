use burn::backend::wgpu::Wgpu;
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate a test image
    let w = 128;
    let h = 128;
    let flat_data = vec![0.5f32; 3 * h * w];
    let tensor_data = TensorData::new(flat_data, [3, h, w]);
    let tensor = Tensor::<Backend, 3>::from_data(tensor_data, &device);
    let image = Image::new(tensor);

    // 2. Perform blurs
    println!("Applying Box Blur...");
    let box_blurred = image.clone().box_blur(5)?;
    box_blurred.save("output_box_blur.png")?;

    println!("Applying Gaussian Blur...");
    let gaussian_blurred = image.clone().gaussian_blur(5, 1.0)?;
    gaussian_blurred.save("output_gaussian_blur.png")?;

    println!("Applying Median Blur...");
    let median_blurred = image.clone().median_blur(5)?;
    median_blurred.save("output_median_blur.png")?;

    println!("Applying Bilateral Filter...");
    let bilateral_filtered = image.clone().bilateral_filter(5, 0.1, 10.0)?;
    bilateral_filtered.save("output_bilateral.png")?;

    println!("Applying Separable Filter...");
    let sep_filtered = image
        .clone()
        .sep_filter_2d(&[0.2, 0.6, 0.2], &[0.2, 0.6, 0.2])?;
    sep_filtered.save("output_sep_filter.png")?;

    println!("Filter operations example completed successfully.");
    Ok(())
}
