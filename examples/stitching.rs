// Demonstrates image stitching (panorama creation) from multiple images.
// Loads two real images and stitches them together.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load two real images for stitching
    let left: Image<Backend> = Image::open("assets/images/gradient.png", &device)?;
    let right: Image<Backend> = Image::open("assets/images/color_bars.png", &device)?;

    println!(
        "Left: {}x{}, Right: {}x{}",
        left.width(),
        left.height(),
        right.width(),
        right.height()
    );

    // Stitch into panorama
    println!("Stitching images together...");
    let stitcher = Stitcher;
    let panorama = stitcher.stitch(&[left, right])?;

    println!("Stitched panorama shape: {:?}", panorama.shape());
    panorama.save("output_stitching.png")?;
    println!("Saved panorama to 'output_stitching.png'");

    Ok(())
}
