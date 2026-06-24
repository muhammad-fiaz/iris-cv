use burn::backend::wgpu::Wgpu;
use observers::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Create a dummy model path (simulate loading a YOLO ONNX model)
    let model_path = "yolo_mock.onnx";

    // Instantiate OnnxModel
    let model = OnnxModel::<Backend>::load(model_path, &device)?;
    println!("Loaded model: {}", model.model_path);

    // 2. Load or generate a test image
    let w = 640;
    let h = 480;
    let flat_data = vec![0.5f32; 3 * h * w];
    let tensor_data = TensorData::new(flat_data, [3, h, w]);
    let tensor = Tensor::<Backend, 3>::from_data(tensor_data, &device);
    let image = Image::new(tensor);

    // 3. Preprocess the image to input tensor [1, C, H, W]
    let input_tensor = model.preprocess(&image)?;
    println!("Input tensor shape: {:?}", input_tensor.dims());

    // 4. Run prediction
    // Predicts a mock output tensor of shape [1, 10, 6]
    let output: Tensor<Backend, 3> = model.predict_raw(input_tensor)?;
    println!("Output prediction tensor shape: {:?}", output.dims());

    Ok(())
}
