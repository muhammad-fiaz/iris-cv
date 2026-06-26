---
title: "Color Processing Example"
description: "Color space conversions, CLAHE, histogram equalization, and in-range thresholding with Iris."
keywords: ["color processing", "color spaces", "CLAHE", "histogram"]
---

# Color Processing

Demonstrates color space conversions (HSV, LAB, YUV, YCrCb, HLS, XYZ), CLAHE, histogram equalization, and in-range color thresholding.

```bash
cargo run --example color_processing --features wgpu
```

## Source

```rust
// Demonstrates color space conversions, CLAHE, histogram equalization,
// and in-range color thresholding on a real image.
//
// Supported color spaces: HSV, LAB, YUV, YCrCb, HLS, XYZ.
// Operations: CLAHE adaptive equalization, color histogram equalization,
// in-range masking, channel splitting/merging, and roundtrip conversions.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // Load a real color image
    let img: Image<Backend> = Image::open("assets/images/color_bars.png", &device)?;
    println!(
        "Loaded image: {}x{} ({} channels)",
        img.width(),
        img.height(),
        img.channels()
    );

    // --- 1. RGB to HSV conversion ---
    println!("\n--- RGB to HSV ---");
    let hsv = img.rgb_to_hsv()?;
    println!("HSV shape: {:?}", hsv.shape());
    hsv.save("output_color_processing_hsv.png")?;

    // Verify roundtrip: HSV -> RGB
    let rgb_back = hsv.hsv_to_rgb()?;
    println!("HSV -> RGB roundtrip shape: {:?}", rgb_back.shape());

    // --- 2. RGB to LAB conversion ---
    println!("\n--- RGB to LAB ---");
    let lab = img.rgb_to_lab()?;
    println!("LAB shape: {:?}", lab.shape());
    lab.save("output_color_processing_lab.png")?;

    // Verify roundtrip: LAB -> RGB
    let rgb_from_lab = lab.lab_to_rgb()?;
    println!("LAB -> RGB roundtrip shape: {:?}", rgb_from_lab.shape());

    // --- 3. RGB to YUV conversion ---
    println!("\n--- RGB to YUV ---");
    let yuv = img.rgb_to_yuv()?;
    println!("YUV shape: {:?}", yuv.shape());
    yuv.save("output_color_processing_yuv.png")?;

    // Verify roundtrip: YUV -> RGB
    let rgb_from_yuv = yuv.yuv_to_rgb()?;
    println!("YUV -> RGB roundtrip shape: {:?}", rgb_from_yuv.shape());

    // --- 4. RGB to YCrCb conversion ---
    println!("\n--- RGB to YCrCb ---");
    let ycrcb = img.rgb_to_ycrcb()?;
    println!("YCrCb shape: {:?}", ycrcb.shape());
    ycrcb.save("output_color_processing_ycrcb.png")?;

    // Verify roundtrip: YCrCb -> RGB
    let rgb_from_ycrcb = ycrcb.ycrcb_to_rgb()?;
    println!("YCrCb -> RGB roundtrip shape: {:?}", rgb_from_ycrcb.shape());

    // --- 5. RGB to HLS conversion ---
    println!("\n--- RGB to HLS ---");
    let hls = img.rgb_to_hls()?;
    println!("HLS shape: {:?}", hls.shape());
    hls.save("output_color_processing_hls.png")?;

    // --- 6. RGB to XYZ conversion ---
    println!("\n--- RGB to XYZ ---");
    let xyz = img.rgb_to_xyz()?;
    println!("XYZ shape: {:?}", xyz.shape());
    xyz.save("output_color_processing_xyz.png")?;

    // --- 7. CLAHE (Adaptive Histogram Equalization) ---
    println!("\n--- CLAHE (Adaptive Histogram Equalization) ---");
    let gray = img.grayscale()?;
    let clahe = gray.clahe(2.0, 4)?;
    println!("CLAHE (clip=2.0, grid=4): shape = {:?}", clahe.shape());
    clahe.save("output_color_processing_clahe.png")?;

    // --- 8. Color Histogram Equalization ---
    println!("\n--- Color Histogram Equalization ---");
    let equalized = img.equalize_hist_color()?;
    println!("Equalized shape: {:?}", equalized.shape());
    equalized.save("output_color_processing_hist_eq.png")?;

    // --- 9. In-Range Color Thresholding ---
    println!("\n--- In-Range Color Thresholding ---");
    let mask = img.in_range(&[0.2, 0.1, 0.1], &[0.9, 0.8, 0.7])?;
    println!("In-range mask shape: {:?}", mask.shape());
    mask.save("output_color_processing_inrange.png")?;

    // --- 10. Channel Splitting and Merging ---
    println!("\n--- Channel Splitting and Merging ---");
    let channels = img.split_channels()?;
    println!("Split into {} channels", channels.len());
    for (i, ch) in channels.iter().enumerate() {
        println!("  Channel {}: shape = {:?}", i, ch.shape());
    }
    let merged = Image::merge_channels(&channels)?;
    println!("Merged back: shape = {:?}", merged.shape());

    println!("\nColor processing example completed. All outputs saved.");
    Ok(())
}
```
