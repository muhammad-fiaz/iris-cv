// Demonstrates ONNX model loading, preprocessing, and raw prediction.
// Uses a mock model path (no real ONNX file required for compilation).

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Load a mock ONNX model
    let model_path = "yolo_mock.onnx";
    let model = OnnxModel::<Backend>::load(model_path, &device)?;
    println!("Loaded model: {}", model.model_path);

    // 2. Load a real image for inference
    let image: Image<Backend> = Image::open("assets/images/gradient.png", &device)?;
    println!(
        "Input image: {}x{} ({} channels)",
        image.width(),
        image.height(),
        image.channels()
    );

    // 3. Preprocess to [1, C, H, W] tensor
    let input_tensor = model.preprocess(&image)?;
    println!("Input tensor shape: {:?}", input_tensor.dims());

    // 4. Run prediction (returns mock output [1, 10, 6])
    let output: Tensor<Backend, 3> = model.predict_raw(input_tensor)?;
    println!("Output prediction shape: {:?}", output.dims());

    Ok(())
}
