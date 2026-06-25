// Demonstrates morphological operations: dilation, erosion, opening, and closing.
// Loads a real image and applies each morphological operation.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load a real image
    let image: Image<Backend> = Image::open("assets/images/checkerboard.png", &device)?;
    println!(
        "Loaded image: {}x{} ({} channels)",
        image.width(),
        image.height(),
        image.channels()
    );

    // Structuring element info
    let element = Morphology::get_structuring_element(MorphShape::Rect, Size::new(5, 5));
    println!(
        "Structuring element Rect 5x5: {}x{}",
        element.len(),
        element[0].len()
    );

    // Dilation
    println!("Applying Dilation (kernel size 3)...");
    let dilated = image.clone().dilate(3)?;
    dilated.save("output_morph_dilation.png")?;

    // Erosion
    println!("Applying Erosion (kernel size 3)...");
    let eroded = image.clone().erode(3)?;
    eroded.save("output_morph_erosion.png")?;

    // Opening (erosion then dilation)
    println!("Applying Morphology Ex (Opening)...");
    let opened = image.clone().morphology_ex(MorphOp::Opening, 3)?;
    opened.save("output_morph_opening.png")?;

    // Closing (dilation then erosion)
    println!("Applying Morphology Ex (Closing)...");
    let closed = image.clone().morphology_ex(MorphOp::Closing, 3)?;
    closed.save("output_morph_closing.png")?;

    println!("All morphology operations completed. Outputs saved to output_morph_*.png");
    Ok(())
}
