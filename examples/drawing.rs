// Demonstrates drawing shapes, text, and annotations on images.
// Creates a canvas and draws lines, rectangles, circles, and text.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Start with a real image as the canvas
    let mut canvas: Image<Backend> = Image::open("assets/images/gradient.png", &device)?;

    // Draw a red line across the top
    canvas = canvas.draw_line(
        Point::new(10, 10),
        Point::new(630, 10),
        Scalar::new(1.0, 0.0, 0.0, 0.0),
    )?;

    // Draw a green border rectangle
    canvas = canvas.draw_rectangle(
        Rect::new(50, 50, 200, 150),
        Scalar::new(0.0, 1.0, 0.0, 0.0),
        3,
    )?;

    // Draw a blue filled circle
    canvas = canvas.draw_circle(
        Point::new(400, 200),
        60,
        Scalar::new(0.0, 0.0, 1.0, 0.0),
        -1,
    )?;

    // Render white text label
    canvas = canvas.draw_text(
        "Iris CV",
        Point::new(50, 250),
        2,
        Scalar::new(1.0, 1.0, 1.0, 0.0),
    )?;

    canvas.save("output_drawing.png")?;
    println!("Saved drawing output to 'output_drawing.png'");

    Ok(())
}
