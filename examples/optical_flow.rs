use burn::backend::wgpu::Wgpu;
use observers::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate two mock frames
    let w = 100;
    let h = 100;
    let flat = vec![0.5f32; 3 * h * w];
    let img1 = Image::new(Tensor::<Backend, 3>::from_data(TensorData::new(flat.clone(), [3, h, w]), &device));
    let img2 = Image::new(Tensor::<Backend, 3>::from_data(TensorData::new(flat, [3, h, w]), &device));

    // 2. Dense Optical Flow (Farneback)
    println!("Calculating dense optical flow (Farneback)...");
    let flow = OpticalFlow::calc_dense_farneback(&img1, &img2)?;
    println!("Dense flow tensor shape: {:?}", flow.dims());

    // 3. Sparse Optical Flow (Lucas-Kanade)
    println!("Calculating sparse optical flow (Lucas-Kanade)...");
    let prev_pts = vec![Point::new(10.0, 10.0), Point::new(20.0, 20.0)];
    let (next_pts, status) = OpticalFlow::calc_sparse_pyr_lk(&img1, &img2, &prev_pts)?;
    for i in 0..prev_pts.len() {
        if status[i] == 1 {
            println!("Point tracked from {:?} to {:?}", prev_pts[i], next_pts[i]);
        }
    }

    println!("Optical flow example completed successfully.");
    Ok(())
}
