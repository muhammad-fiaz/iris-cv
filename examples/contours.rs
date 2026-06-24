use burn::backend::wgpu::Wgpu;
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate a test image with a high-contrast square in the center
    let w = 100;
    let h = 100;
    let mut flat_data = vec![0.0f32; 3 * h * w];
    for y in 30..70 {
        for x in 30..70 {
            flat_data[y * w + x] = 1.0;
            flat_data[h * w + y * w + x] = 1.0;
            flat_data[2 * h * w + y * w + x] = 1.0;
        }
    }
    let tensor_data = TensorData::new(flat_data, [3, h, w]);
    let tensor = Tensor::<Backend, 3>::from_data(tensor_data, &device);
    let image = Image::new(tensor);

    // 2. Find contours
    let contours = image.find_contours()?;
    println!("Found {} contour(s)", contours.len());

    if let Some(contour) = contours.first() {
        println!("Contour size: {} points", contour.points.len());

        // Compute convex hull
        let hull = contour.convex_hull();
        println!("Convex hull size: {} points", hull.points.len());

        // Compute moments
        let m = contour.moments();
        println!("Contour Area (m00): {}", m.m00);
        if let Some(centroid) = m.centroid() {
            println!("Contour Centroid: ({:.2}, {:.2})", centroid.x, centroid.y);
        }
    }

    Ok(())
}
