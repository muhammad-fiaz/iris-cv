use burn::backend::wgpu::Wgpu;
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // --- Demonstrate Modern user-friendly Gui Struct ---
    println!("\n--- Modern Gui Interface ---");
    let winname_gui = "Gui Window";
    Gui::named_window(winname_gui)?;

    let w = 320;
    let h = 240;
    let flat_data = vec![0.5f32; 3 * h * w];
    let tensor = Tensor::<Backend, 3>::from_data(TensorData::new(flat_data, [3, h, w]), &device);
    let frame = Image::new(tensor);

    // Render frame
    Gui::imshow(winname_gui, &frame)?;

    // Create trackbar & buttons
    Gui::create_trackbar("Alpha", winname_gui, 50, 100)?;
    Gui::set_trackbar_pos("Alpha", winname_gui, 75)?;
    let val = Gui::get_trackbar_pos("Alpha", winname_gui)?;
    println!("Gui 'Alpha' Trackbar position: {}", val);

    // Register mouse callback
    Gui::set_mouse_callback(winname_gui, |event, x, y, flags| {
        println!(
            "[Gui Callback] Mouse event: {}, x: {}, y: {}, flags: {}",
            event, x, y, flags
        );
    })?;

    Gui::wait_key(100)?;
    Gui::destroy_all_windows()?;

    let preview_path_gui = format!("preview_{}.png", winname_gui.replace(" ", "_"));
    let _ = std::fs::remove_file(preview_path_gui);

    Ok(())
}
