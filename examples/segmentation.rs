use burn::backend::wgpu::Wgpu;
use observers::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate an image
    let w = 128;
    let h = 128;
    let flat_data = vec![0.5f32; 3 * h * w];
    let img = Image::new(Tensor::<Backend, 3>::from_data(TensorData::new(flat_data, [3, h, w]), &device));

    // 2. Perform semantic segmentation
    println!("Instantiating Segmenter...");
    let segmenter = Segmenter::<Backend>::default();
    let mask = segmenter.segment(&img)?;
    println!("Output segmentation mask shape: {:?}", mask.mask.dims());

    // 3. Connected components labeling & stats
    println!("Running connected components labeling on mock binary image...");
    let mut binary_data = vec![0.0f32; 1 * 50 * 50];
    // Create a 5x5 white square in center
    for y in 20..25 {
        for x in 20..25 {
            binary_data[y * 50 + x] = 1.0;
        }
    }
    let binary_img = Image::new(Tensor::<Backend, 3>::from_data(TensorData::new(binary_data, [1, 50, 50]), &device));
    let (labels, stats) = binary_img.connected_components_with_stats()?;

    println!("Found {} connected component(s):", stats.len());
    for stat in stats {
        println!(" - Component label: {}", stat.label);
        println!(" - Bounding box: {:?}", stat.bbox);
        println!(" - Area: {}", stat.area);
        println!(" - Centroid: {:?}", stat.centroid);
    }

    Ok(())
}
