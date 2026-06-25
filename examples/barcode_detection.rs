// Demonstrates barcode detection using the BarcodeDetector.
// Loads a real image and attempts to detect and decode barcodes.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load a real image to search for barcodes
    let image: Image<Backend> = Image::open("assets/images/test_pattern.png", &device)?;

    // Detect and decode barcodes
    let detector = BarcodeDetector::new();
    let barcodes = detector.detect_and_decode(&image)?;

    println!("Detected {} barcode(s):", barcodes.len());
    for bc in &barcodes {
        println!("  - Format: '{}', payload: '{}'", bc.format, bc.payload);
        println!("    Corners: {:?}", bc.corners);
    }

    if barcodes.is_empty() {
        println!("  (No barcodes found in test_pattern — expected)");
    }

    image.save("output_barcode_detection.png")?;
    println!("Saved input image to 'output_barcode_detection.png'");

    Ok(())
}
