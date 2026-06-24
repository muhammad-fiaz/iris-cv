use burn::backend::wgpu::Wgpu;
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate an empty black canvas image
    let w = 400;
    let h = 300;
    let flat_canvas = vec![0.0f32; 3 * h * w];
    let tensor = Tensor::<Backend, 3>::from_data(TensorData::new(flat_canvas, [3, h, w]), &device);
    let mut canvas = Image::new(tensor);

    // 2. Draw various items
    println!("Drawing shapes on the canvas...");
    // Draw a red line
    canvas = canvas.draw_line(
        Point::new(10, 10),
        Point::new(390, 10),
        Scalar::new(1.0, 0.0, 0.0, 0.0),
    )?;

    // Draw a green border rectangle
    canvas = canvas.draw_rectangle(
        Rect::new(50, 50, 100, 100),
        Scalar::new(0.0, 1.0, 0.0, 0.0),
        2,
    )?;

    // Draw a blue filled circle
    canvas = canvas.draw_circle(
        Point::new(300, 100),
        40,
        Scalar::new(0.0, 0.0, 1.0, 0.0),
        -1,
    )?;

    // Render white text label
    println!("Rendering text label...");
    canvas = canvas.draw_text(
        "Iris library",
        Point::new(50, 200),
        2,
        Scalar::new(1.0, 1.0, 1.0, 0.0),
    )?;

    canvas.save("output_drawing.png")?;
    println!("Saved drawing output to 'output_drawing.png'");

    println!("Drawing example completed successfully.");
    Ok(())
}
