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
    let w = 64;
    let h = 64;
    let flat_data = vec![0.5f32; 3 * h * w];
    let tensor_data = TensorData::new(flat_data, [3, h, w]);
    let tensor = Tensor::<Backend, 3>::from_data(tensor_data, &device);
    let image = Image::new(tensor);

    // 2. Structuring Element (demonstrating the function exists)
    let element = Morphology::get_structuring_element(MorphShape::Rect, Size::new(5, 5));
    println!(
        "Structuring element Rect 5x5: {}x{}",
        element.len(),
        element[0].len()
    );

    // 3. Perform morphology using actual iris API
    println!("Applying Dilation (kernel size 3)...");
    let dilated = image.clone().dilate(3)?;
    dilated.save("output_dilation.png")?;

    println!("Applying Erosion (kernel size 3)...");
    let eroded = image.clone().erode(3)?;
    eroded.save("output_erosion.png")?;

    println!("Applying Morphology Ex (Opening)...");
    let opened = image.clone().morphology_ex(MorphOp::Opening, 3)?;
    opened.save("output_opening.png")?;

    println!("Applying Morphology Ex (Closing)...");
    let closed = image.clone().morphology_ex(MorphOp::Closing, 3)?;
    closed.save("output_closing.png")?;

    println!("Morphology operations example completed successfully.");
    Ok(())
}
