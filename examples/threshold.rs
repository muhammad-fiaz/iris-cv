// Demonstrates various thresholding operations: binary, truncation, to-zero, and Otsu.
// Loads a real grayscale-compatible image and applies each threshold type.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load a real image and convert to grayscale for thresholding
    let image: Image<Backend> = Image::open("assets/images/gradient.png", &device)?.grayscale()?;
    println!(
        "Loaded grayscale image: {}x{}",
        image.width(),
        image.height()
    );

    // Binary Threshold
    println!("Applying Binary Threshold...");
    let binary = image.clone().threshold(0.5, 1.0, ThresholdType::Binary)?;
    binary.save("output_thresh_binary.png")?;

    // Truncate Threshold
    println!("Applying Truncate Threshold...");
    let trunc = image.clone().threshold(0.5, 1.0, ThresholdType::Trunc)?;
    trunc.save("output_thresh_trunc.png")?;

    // ToZero Threshold
    println!("Applying ToZero Threshold...");
    let tozero = image.clone().threshold(0.3, 1.0, ThresholdType::ToZero)?;
    tozero.save("output_thresh_tozero.png")?;

    // Otsu Thresholding
    println!("Applying Otsu Threshold...");
    let otsu = image.clone().threshold_otsu(1.0)?;
    otsu.save("output_thresh_otsu.png")?;

    println!("All thresholding operations completed. Outputs saved to output_thresh_*.png");
    Ok(())
}
