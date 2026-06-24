use burn::backend::wgpu::Wgpu;
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate 2D points data for clustering
    // Let's create two clusters: one around (1.0, 1.0) and one around (10.0, 10.0)
    let raw_points = vec![
        1.0f32, 1.2, 0.9, 1.1, 1.1, 0.9, 10.0, 10.5, 9.8, 10.2, 10.2, 9.8,
    ];
    let data = Tensor::<Backend, 2>::from_data(TensorData::new(raw_points, [6, 2]), &device);

    // 2. Perform K-Means clustering with K=2
    println!("Fitting K-Means model (K=2, max iterations 10)...");
    let mut km = KMeans::new(2, 10);
    km.fit(&data)?;

    if let Some(ref centroids) = km.centroids {
        println!("Calculated centroids:");
        println!("{:?}", centroids.clone().into_data());
    }

    // 3. Predict cluster assignments
    let assignments = km.predict(&data)?;
    println!(
        "Point assignments to clusters: {:?}",
        assignments.into_data()
    );

    println!("K-Means clustering example completed successfully.");
    Ok(())
}
