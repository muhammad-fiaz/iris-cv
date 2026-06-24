use burn::backend::wgpu::Wgpu;
use observers::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate empty image
    let w = 500;
    let h = 500;
    let flat_data = vec![0.5f32; 3 * h * w];
    let tensor_data = TensorData::new(flat_data, [3, h, w]);
    let tensor = Tensor::<Backend, 3>::from_data(tensor_data, &device);
    let image = Image::new(tensor);

    // 2. Instantiate QR detector and search QR codes
    let detector = QrDetector::new();
    let qr_codes = detector.detect_and_decode(&image)?;

    println!("Detected {} QR code(s):", qr_codes.len());
    for qr in qr_codes {
        println!(" - Payload: '{}'", qr.payload);
        println!(" - Corners: {:?}", qr.corners);
    }

    Ok(())
}
