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
    let w = 400;
    let h = 100;
    let flat_data = vec![0.5f32; 3 * h * w];
    let img = Image::new(Tensor::<Backend, 3>::from_data(TensorData::new(flat_data, [3, h, w]), &device));

    // 2. Perform OCR text recognition
    println!("Running OCR text recognition...");
    let ocr = OcrPipeline::default();
    let ocr_results = ocr.recognize(&img)?;

    println!("OCR Results:");
    for result in ocr_results {
        println!(" - Text: '{}'", result.text);
        println!(" - Bounding box: {:?}", result.bbox);
        println!(" - Confidence: {}", result.confidence);
    }

    Ok(())
}
