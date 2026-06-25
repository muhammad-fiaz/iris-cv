use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::core::{Point, Scalar};
use iris::drawing::MarkerType;
use iris::image::Image;
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!("=== Iris New Features Demo ===\n");

    // --- 1. Color Space Conversions ---
    println!("--- Color Space Conversions ---");
    let data = vec![
        0.8f32, 0.2, 0.5, 0.1, 0.9, 0.3, 0.6, 0.4, 0.7, 0.2, 0.8, 0.1,
    ];
    let rgb = Image::new(Tensor::<Backend, 3>::from_data(
        TensorData::new(data, [3, 2, 2]),
        &device,
    ));

    let xyz = rgb.rgb_to_xyz()?;
    println!("  RGB -> XYZ: shape = {:?}", xyz.shape());

    let lab = rgb.rgb_to_lab()?;
    println!("  RGB -> LAB: shape = {:?}", lab.shape());

    let yuv = rgb.rgb_to_yuv()?;
    println!("  RGB -> YUV: shape = {:?}", yuv.shape());

    let ycrcb = rgb.rgb_to_ycrcb()?;
    println!("  RGB -> YCrCb: shape = {:?}", ycrcb.shape());

    let back = xyz.xyz_to_rgb()?;
    println!("  XYZ -> RGB roundtrip: shape = {:?}", back.shape());

    // --- 2. Noise Generation ---
    println!("\n--- Noise Generation ---");
    let base = Image::new(Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.5f32; 3 * 32 * 32], [3, 32, 32]),
        &device,
    ));

    let gaussian = base.add_gaussian_noise(0.0, 0.05)?;
    println!("  Gaussian noise: shape = {:?}", gaussian.shape());

    let salt_pepper = base.add_salt_pepper_noise(0.05)?;
    println!("  Salt-pepper noise: shape = {:?}", salt_pepper.shape());

    let speckle = base.add_speckle_noise(0.1)?;
    println!("  Speckle noise: shape = {:?}", speckle.shape());

    // --- 3. General Convolution (filter2D) ---
    println!("\n--- Filter2D (General Convolution) ---");
    let kernel: Vec<&[f32]> = vec![&[-1.0, -1.0, -1.0], &[-1.0, 8.0, -1.0], &[-1.0, -1.0, -1.0]];
    let edges = base.filter2d(&kernel, None, 0.0)?;
    println!("  Laplacian kernel: shape = {:?}", edges.shape());

    // --- 4. Alpha Blending (addWeighted) ---
    println!("\n--- Alpha Blending (addWeighted) ---");
    let blended = base.add_weighted(&gaussian, 0.7, 0.3, 0.0)?;
    println!(
        "  Blended (0.7*src + 0.3*noise): shape = {:?}",
        blended.shape()
    );

    // --- 5. CLAHE ---
    println!("\n--- CLAHE (Adaptive Histogram Equalization) ---");
    let gray = base.grayscale()?;
    let clahe = gray.clahe(2.0, 4)?;
    println!("  CLAHE (clip=2.0, grid=4): shape = {:?}", clahe.shape());

    // --- 6. LUT ---
    println!("\n--- Lookup Table (LUT) ---");
    let mut lut = [0.0f32; 256];
    for (i, item) in lut.iter_mut().enumerate() {
        *item = 1.0 - (i as f32 / 255.0);
    }
    let inverted = base.apply_lut(&lut)?;
    println!("  Inverted via LUT: shape = {:?}", inverted.shape());

    // --- 7. inRange (Color Thresholding) ---
    println!("\n--- inRange (Color Thresholding) ---");
    let mask = rgb.in_range(&[0.2, 0.1, 0.1], &[0.9, 0.8, 0.7])?;
    println!("  inRange mask: shape = {:?}", mask.shape());

    // --- 8. Normalize ---
    println!("\n--- Normalize ---");
    let normalized = base.normalize(0.0, 1.0)?;
    println!("  Normalized [0,1]: shape = {:?}", normalized.shape());

    // --- 9. Drawing Extras ---
    println!("\n--- Drawing Extras ---");
    let canvas = Image::new(Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.0f32; 3 * 100 * 100], [3, 100, 100]),
        &device,
    ));

    let canvas = canvas.draw_ellipse(
        Point::new(50, 50),
        (30, 20),
        30.0,
        0.0,
        360.0,
        Scalar::all(1.0),
        1,
    )?;
    println!("  Ellipse drawn: shape = {:?}", canvas.shape());

    let canvas = canvas.draw_polyline(
        &[
            Point::new(10, 80),
            Point::new(30, 60),
            Point::new(50, 80),
            Point::new(70, 60),
            Point::new(90, 80),
        ],
        Scalar::all(0.5),
        1,
    )?;
    println!("  Polyline drawn: shape = {:?}", canvas.shape());

    let canvas = canvas.fill_poly(
        &[Point::new(10, 10), Point::new(30, 10), Point::new(20, 30)],
        Scalar::all(0.8),
    )?;
    println!("  Filled polygon: shape = {:?}", canvas.shape());

    let canvas = canvas.draw_arrowed_line(
        Point::new(80, 10),
        Point::new(10, 10),
        Scalar::all(1.0),
        1,
        0.3,
    )?;
    println!("  Arrowed line: shape = {:?}", canvas.shape());

    let canvas = canvas.draw_marker(Point::new(50, 50), Scalar::all(0.5), MarkerType::Circle, 8)?;
    println!("  Circle marker: shape = {:?}", canvas.shape());

    // --- 10. Custom Morphological Kernels ---
    println!("\n--- Custom Morph Kernels ---");
    let cross_kernel: Vec<&[u8]> = vec![&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]];
    let dilated = base.clone().dilate_with_kernel(&cross_kernel)?;
    println!("  Dilate with cross: shape = {:?}", dilated.shape());

    let eroded = base.clone().erode_with_kernel(&cross_kernel)?;
    println!("  Erode with cross: shape = {:?}", eroded.shape());

    // --- 11. Convert Scale Abs ---
    println!("\n--- Convert Scale Abs ---");
    let abs = base.convert_scale_abs(2.0, -0.5)?;
    println!("  convert_scale_abs(2.0, -0.5): shape = {:?}", abs.shape());

    // --- 12. Histogram Comparison ---
    println!("\n--- Histogram Comparison ---");
    let hist1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let hist2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let hist3 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let corr_same = Image::<Backend>::compare_hist(&hist1, &hist2, "correlation")?;
    let corr_diff = Image::<Backend>::compare_hist(&hist1, &hist3, "correlation")?;
    println!("  Identical histograms correlation: {corr_same:.4}");
    println!("  Reversed histograms correlation:  {corr_diff:.4}");

    // --- 13. convert_scale_abs save ---
    println!("\n--- Saving Outputs ---");
    let demo_img = Image::new(Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.5f32; 3 * 64 * 64], [3, 64, 64]),
        &device,
    ));
    let noisy = demo_img.add_gaussian_noise(0.0, 0.1)?;
    let denoised = noisy.clone().gaussian_blur(5, 1.0)?;
    denoised.save("output_denoised.png")?;
    println!("  Saved denoised image to output_denoised.png");

    println!("\n=== All demos completed successfully! ===");
    Ok(())
}
