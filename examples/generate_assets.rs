use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Tensor, TensorData};
use iris::prelude::*;

type Backend = Wgpu;

fn main() -> Result<()> {
    let device = WgpuDevice::default();

    println!("Generating sample assets for iris CV library examples...\n");

    // Ensure output directories exist
    std::fs::create_dir_all("assets/images").ok();
    std::fs::create_dir_all("assets/videos").ok();

    // 1. Color gradient image (640x480)
    println!("Generating gradient.png (640x480 color gradient)...");
    generate_gradient(&device)?;

    // 2. Checkerboard pattern (640x480)
    println!("Generating checkerboard.png (640x480 checkerboard)...");
    generate_checkerboard(&device)?;

    // 3. Test pattern with circles and lines (640x480)
    println!("Generating test_pattern.png (640x480 test pattern)...");
    generate_test_pattern(&device)?;

    // 4. Grayscale gradient (320x240)
    println!("Generating gray_gradient.png (320x240 grayscale gradient)...");
    generate_gray_gradient(&device)?;

    // 5. Color bars image (640x480)
    println!("Generating color_bars.png (640x480 color bars)...");
    generate_color_bars(&device)?;

    println!("\nAll assets generated successfully in assets/images/");
    Ok(())
}

/// Generates a 640x480 color gradient image.
/// Each pixel's color varies smoothly across the image.
fn generate_gradient(device: &WgpuDevice) -> Result<()> {
    let w = 640;
    let h = 480;

    let mut data = vec![0.0f32; 3 * h * w];

    for y in 0..h {
        for x in 0..w {
            let r = x as f32 / w as f32;
            let g = y as f32 / h as f32;
            let b = 1.0 - (x as f32 / w as f32);
            let idx = y * w + x;
            data[idx] = r;
            data[h * w + idx] = g;
            data[2 * h * w + idx] = b;
        }
    }

    let tensor = Tensor::<Backend, 3>::from_data(TensorData::new(data, [3, h, w]), device);
    let img = Image::new(tensor);
    img.save("assets/images/gradient.png")?;
    Ok(())
}

/// Generates a 640x480 checkerboard pattern.
fn generate_checkerboard(device: &WgpuDevice) -> Result<()> {
    let w = 640;
    let h = 480;
    let tile_size = 64;

    let mut data = vec![0.0f32; 3 * h * w];

    for y in 0..h {
        for x in 0..w {
            let tile_x = x / tile_size;
            let tile_y = y / tile_size;
            let is_white = (tile_x + tile_y) % 2 == 0;
            let val = if is_white { 1.0f32 } else { 0.0f32 };
            let idx = y * w + x;
            data[idx] = val;
            data[h * w + idx] = val;
            data[2 * h * w + idx] = val;
        }
    }

    let tensor = Tensor::<Backend, 3>::from_data(TensorData::new(data, [3, h, w]), device);
    let img = Image::new(tensor);
    img.save("assets/images/checkerboard.png")?;
    Ok(())
}

/// Generates a 640x480 test pattern with concentric circles, crosshairs, and border lines.
fn generate_test_pattern(device: &WgpuDevice) -> Result<()> {
    let w = 640;
    let h = 480;
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;

    let mut data = vec![0.0f32; 3 * h * w];

    for y in 0..h {
        for x in 0..w {
            let fx = x as f32;
            let fy = y as f32;

            // Distance from center
            let dist = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt();
            let max_dist = ((cx.powi(2) + cy.powi(2)).sqrt()).min(300.0);

            // Concentric circles
            let circle_interval = 40.0;
            let in_circle = (dist % circle_interval) < 2.0;
            let circle_val = if in_circle && dist < max_dist {
                1.0f32
            } else {
                0.0f32
            };

            // Crosshairs
            let cross_width = 2;
            let on_hcross = (fy - cy).abs() < cross_width as f32;
            let on_vcross = (fx - cx).abs() < cross_width as f32;
            let cross_val = if (on_hcross || on_vcross) && dist < max_dist {
                0.8f32
            } else {
                0.0f32
            };

            // Diagonal lines
            let on_diag1 = (fx - fy).abs() < 2.0;
            let on_diag2 = ((fx - cx) + (fy - cy)).abs() < 2.0;
            let diag_val = if (on_diag1 || on_diag2) && dist < max_dist {
                0.6f32
            } else {
                0.0f32
            };

            // Border rectangle
            let border_margin = 40.0;
            let on_border = fx < border_margin
                || fx >= w as f32 - border_margin
                || fy < border_margin
                || fy >= h as f32 - border_margin;
            let border_val = if on_border { 1.0f32 } else { 0.0f32 };

            // Combine layers
            let r = (circle_val + border_val).min(1.0);
            let g = (cross_val + diag_val).min(1.0);
            let b = (circle_val * 0.5 + cross_val * 0.5 + diag_val * 0.5).min(1.0);

            let idx = y * w + x;
            data[idx] = r;
            data[h * w + idx] = g;
            data[2 * h * w + idx] = b;
        }
    }

    let tensor = Tensor::<Backend, 3>::from_data(TensorData::new(data, [3, h, w]), device);
    let img = Image::new(tensor);
    img.save("assets/images/test_pattern.png")?;
    Ok(())
}

/// Generates a 320x240 grayscale gradient image.
fn generate_gray_gradient(device: &WgpuDevice) -> Result<()> {
    let w = 320;
    let h = 240;

    let mut data = vec![0.0f32; w * h];

    for y in 0..h {
        for x in 0..w {
            // Radial gradient from center
            let cx = w as f32 / 2.0;
            let cy = h as f32 / 2.0;
            let fx = x as f32;
            let fy = y as f32;
            let dist = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt();
            let max_dist = (cx.powi(2) + cy.powi(2)).sqrt();
            let val = (1.0 - dist / max_dist).clamp(0.0, 1.0);
            data[y * w + x] = val;
        }
    }

    // Grayscale image has shape [1, H, W]
    let tensor = Tensor::<Backend, 3>::from_data(TensorData::new(data, [1, h, w]), device);
    let img = Image::new(tensor);
    img.save("assets/images/gray_gradient.png")?;
    Ok(())
}

/// Generates a 640x480 color bars image (SMPTE-style).
fn generate_color_bars(device: &WgpuDevice) -> Result<()> {
    let w = 640;
    let h = 480;

    // Standard 7 color bars: white, yellow, cyan, green, magenta, red, blue
    #[rustfmt::skip]
    let color_bars: Vec<(f32, f32, f32)> = vec![
        (1.0, 1.0, 1.0), // white
        (1.0, 1.0, 0.0), // yellow
        (0.0, 1.0, 1.0), // cyan
        (0.0, 1.0, 0.0), // green
        (1.0, 0.0, 1.0), // magenta
        (1.0, 0.0, 0.0), // red
        (0.0, 0.0, 1.0), // blue
    ];

    let bar_width = w / color_bars.len();

    let mut data = vec![0.0f32; 3 * h * w];

    for y in 0..h {
        for x in 0..w {
            let bar_idx = (x / bar_width).min(color_bars.len() - 1);
            let (r, g, b) = color_bars[bar_idx];

            let idx = y * w + x;
            data[idx] = r;
            data[h * w + idx] = g;
            data[2 * h * w + idx] = b;
        }
    }

    let tensor = Tensor::<Backend, 3>::from_data(TensorData::new(data, [3, h, w]), device);
    let img = Image::new(tensor);
    img.save("assets/images/color_bars.png")?;
    Ok(())
}
