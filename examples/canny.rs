use burn::backend::wgpu::Wgpu;
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate a test image with a high-contrast square in the center
    let w = 400;
    let h = 400;
    let mut flat_data = vec![0.1f32; 3 * h * w];
    for y in 100..300 {
        for x in 100..300 {
            flat_data[y * w + x] = 0.9;
            flat_data[h * w + y * w + x] = 0.9;
            flat_data[2 * h * w + y * w + x] = 0.9;
        }
    }
    let tensor_data = TensorData::new(flat_data, [3, h, w]);
    let tensor = Tensor::<Backend, 3>::from_data(tensor_data, &device);
    let image = Image::new(tensor);

    // 2. Perform Canny edge detection
    let edges = image.canny(0.1, 0.4)?;

    println!("Detected edges image shape: {:?}", edges.shape());

    // 3. Save the edges output
    edges.save("output_canny_edges.png")?;
    println!("Saved Canny edges output to 'output_canny_edges.png'");

    Ok(())
}
