use burn::backend::wgpu::Wgpu;
use observers::prelude::*;

fn main() -> Result<()> {
    // 1. Select compute backend
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 2. Open an image
    // Note: In real scenarios, pass a valid image path like "cat.jpg".
    // Since we want this example to run without failing on missing assets, we generate a template image first.
    let w = 800;
    let h = 600;
    let flat_data = vec![0.8f32; 3 * h * w];
    let tensor_data = TensorData::new(flat_data, [3, h, w]);
    let tensor = Tensor::<Backend, 3>::from_data(tensor_data, &device);
    let image = Image::new(tensor);

    println!(
        "Original image: {}x{} with {} channels",
        image.width(),
        image.height(),
        image.channels()
    );

    // 3. Process the image using method chaining
    let processed = image
        .resize(400, 300)?
        .grayscale()?
        .to_rgb()?
        .draw_rectangle(
            Rect::new(50, 50, 200, 150),
            Scalar::new(1.0, 0.0, 0.0, 0.0),
            3,
        )?
        .draw_text(
            "Observers CV",
            Point::new(60, 80),
            2,
            Scalar::new(0.0, 1.0, 0.0, 0.0),
        )?;

    println!(
        "Processed image: {}x{} with {} channels",
        processed.width(),
        processed.height(),
        processed.channels()
    );

    // 4. Save the result
    processed.save("output_processed.png")?;
    println!("Saved processed image to 'output_processed.png'");

    Ok(())
}
