use burn::backend::wgpu::Wgpu;
use observers::prelude::*;

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
    let mut flat_data = vec![0.0f32; 1 * h * w];
    for y in 0..h {
        for x in 0..w {
            flat_data[y * w + x] = (x as f32) / (w as f32);
        }
    }
    let tensor_data = TensorData::new(flat_data, [1, h, w]);
    let tensor = Tensor::<Backend, 3>::from_data(tensor_data, &device);
    let image = Image::new(tensor);

    // 2. Perform fixed thresholding
    println!("Applying Binary Threshold...");
    let binary = image.clone().threshold(0.5, 1.0, ThresholdType::Binary)?;
    binary.save("output_thresh_binary.png")?;

    println!("Applying Truncate Threshold...");
    let trunc = image.clone().threshold(0.5, 1.0, ThresholdType::Trunc)?;
    trunc.save("output_thresh_trunc.png")?;

    println!("Applying ToZero Threshold...");
    let tozero = image.clone().threshold(0.3, 1.0, ThresholdType::ToZero)?;
    tozero.save("output_thresh_tozero.png")?;

    // 3. Otsu Thresholding
    println!("Applying Otsu Threshold...");
    let otsu = image.clone().threshold_otsu(1.0)?;
    otsu.save("output_thresh_otsu.png")?;

    println!("Thresholding operations example completed successfully.");
    Ok(())
}
