// Demonstrates K-Means clustering on 2D point data.
// Uses synthetic point clusters (no image file needed).

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = WgpuDevice::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate 2D points: two clusters around (1,1) and (10,10)
    let raw_points = vec![
        1.0f32, 1.2, 0.9, 1.1, 1.1, 0.9, 10.0, 10.5, 9.8, 10.2, 10.2, 9.8,
    ];
    let data = Tensor::<Backend, 2>::from_data(TensorData::new(raw_points, [6, 2]), &device);

    // 2. Fit K-Means with K=2
    println!("Fitting K-Means model (K=2, max_iter=10)...");
    let mut km = KMeans::new(2, 10);
    km.fit(&data)?;

    if let Some(ref centroids) = km.centroids {
        println!("Calculated centroids:");
        println!("  {:?}", centroids.clone().into_data());
    }

    // 3. Predict cluster assignments
    let assignments = km.predict(&data)?;
    println!("Point assignments: {:?}", assignments.into_data());

    println!("K-Means clustering example completed.");
    Ok(())
}
