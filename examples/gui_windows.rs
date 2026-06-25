// Demonstrates the Gui module: creating windows, trackbars, and mouse callbacks.
// Note: requires a display environment; may skip on headless systems.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load a real image to display
    let frame: Image<Backend> = Image::open("assets/images/gradient.png", &device)?;

    // Create a GUI window
    let winname = "Gui Window";
    Gui::named_window(winname)?;

    // Display the image
    Gui::imshow(winname, &frame)?;

    // Create and interact with a trackbar
    Gui::create_trackbar("Alpha", winname, 50, 100)?;
    Gui::set_trackbar_pos("Alpha", winname, 75)?;
    let val = Gui::get_trackbar_pos("Alpha", winname)?;
    println!("Trackbar 'Alpha' position: {}", val);

    // Register mouse callback
    Gui::set_mouse_callback(winname, |event, x, y, flags| {
        println!(
            "[Gui Callback] Mouse event: {}, x: {}, y: {}, flags: {}",
            event, x, y, flags
        );
    })?;

    Gui::wait_key(2000)?;
    Gui::destroy_all_windows()?;

    println!("GUI example completed.");
    Ok(())
}
