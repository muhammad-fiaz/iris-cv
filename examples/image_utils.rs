// Demonstrates image utility functions: filter2D, alpha blending, convert_scale_abs,
// copy_to with mask, normalize, LUT, noise generation, and histogram comparison.
// Loads a real image and applies various utility operations.

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
    let img: Image<Backend> = Image::open("assets/images/gradient.png", &device)?;
    println!(
        "Loaded image: {}x{} ({} channels)",
        img.width(),
        img.height(),
        img.channels()
    );

    // --- 1. filter2D with custom kernels ---
    println!("\n--- Filter2D (Custom Kernel) ---");
    let laplacian_kernel: Vec<&[f32]> = vec![
        &[-1.0, -1.0, -1.0],
        &[-1.0, 8.0, -1.0],
        &[-1.0, -1.0, -1.0],
    ];
    let edges = img.filter2d(&laplacian_kernel, None, 0.0)?;
    println!("Laplacian edges shape: {:?}", edges.shape());
    edges.save("output_image_utils_filter2d.png")?;

    // --- 2. Alpha blending (addWeighted) ---
    println!("\n--- Alpha Blending (addWeighted) ---");
    let blurred = img.clone().gaussian_blur(7, 1.5)?;
    let blended = img.add_weighted(&blurred, 0.7, 0.3, 0.0)?;
    println!("Blended shape: {:?}", blended.shape());
    blended.save("output_image_utils_blend.png")?;

    // --- 3. convert_scale_abs ---
    println!("\n--- Convert Scale Abs ---");
    let scaled = img.convert_scale_abs(2.0, -0.3)?;
    println!("convert_scale_abs(2.0, -0.3): shape = {:?}", scaled.shape());
    scaled.save("output_image_utils_scale_abs.png")?;

    // --- 4. copy_to with mask ---
    println!("\n--- Copy To (with mask) ---");
    let mut dst = img.clone();
    let small = img.resize(200, 150)?;
    // Create a single-channel mask (150x200)
    let mask = Image::new(Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![1.0f32; 150 * 200], [1, 150, 200]),
        &device,
    ));
    // copy_to modifies dst in place where mask is nonzero
    small.copy_to(&mut dst, Some(&mask))?;
    println!("copy_to shape: {:?}", dst.shape());
    dst.save("output_image_utils_copy_to.png")?;

    // --- 5. Normalize ---
    println!("\n--- Normalize ---");
    let normalized = img.normalize(0.0, 1.0)?;
    println!("Normalized shape: {:?}", normalized.shape());
    normalized.save("output_image_utils_normalize.png")?;

    // --- 6. LUT application ---
    println!("\n--- LUT (Lookup Table) ---");
    let mut lut = [0.0f32; 256];
    for (i, item) in lut.iter_mut().enumerate() {
        *item = 1.0 - (i as f32 / 255.0);
    }
    let inverted = img.apply_lut(&lut)?;
    println!("Inverted via LUT: shape = {:?}", inverted.shape());
    inverted.save("output_image_utils_lut.png")?;

    // --- 7. Noise generation ---
    println!("\n--- Noise Generation ---");
    let gaussian = img.add_gaussian_noise(0.0, 0.05)?;
    println!("Gaussian noise: shape = {:?}", gaussian.shape());
    gaussian.save("output_image_utils_noise_gaussian.png")?;

    let salt_pepper = img.add_salt_pepper_noise(0.05)?;
    println!("Salt-pepper noise: shape = {:?}", salt_pepper.shape());
    salt_pepper.save("output_image_utils_noise_sp.png")?;

    let speckle = img.add_speckle_noise(0.1)?;
    println!("Speckle noise: shape = {:?}", speckle.shape());
    speckle.save("output_image_utils_noise_speckle.png")?;

    // --- 8. Histogram computation and comparison ---
    println!("\n--- Histogram Computation ---");
    let hist = img.calc_hist()?;
    println!("Computed {} channel histograms (256 bins each)", hist.len());

    // Compare two histograms from different images
    let hist1 = img.calc_hist()?;
    let blurred_img = img.clone().gaussian_blur(5, 2.0)?;
    let hist2 = blurred_img.calc_hist()?;

    // Compare first channel histograms
    let f32_hist1: Vec<f32> = hist1[0].iter().map(|&x| x as f32).collect();
    let f32_hist2: Vec<f32> = hist2[0].iter().map(|&x| x as f32).collect();

    let corr = Image::<Backend>::compare_hist(&f32_hist1, &f32_hist2, "correlation")?;
    println!("Histogram correlation (original vs blurred): {corr:.4}");

    let chi = Image::<Backend>::compare_hist(&f32_hist1, &f32_hist2, "chi_square")?;
    println!("Histogram chi-square distance: {chi:.4}");

    let inter = Image::<Backend>::compare_hist(&f32_hist1, &f32_hist2, "intersection")?;
    println!("Histogram intersection: {inter:.0}");

    let hel = Image::<Backend>::compare_hist(&f32_hist1, &f32_hist2, "hellinger")?;
    println!("Histogram hellinger distance: {hel:.4}");

    println!("\nImage utils example completed. All outputs saved.");
    Ok(())
}
