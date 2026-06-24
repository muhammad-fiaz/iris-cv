use burn::backend::wgpu::Wgpu;
use iris::prelude::*;

fn main() -> Result<()> {
    type Backend = Wgpu;
    let device = Default::default();

    println!(
        "Using compute backend: {}",
        BurnUtils::backend_name::<Backend>()
    );

    // 1. Generate an image representing a frame containing an ArUco marker
    let w = 400;
    let h = 400;
    let flat_data = vec![0.5f32; 3 * h * w];
    let tensor = Tensor::<Backend, 3>::from_data(TensorData::new(flat_data, [3, h, w]), &device);
    let frame = Image::new(tensor);

    // 2. Detect ArUco markers
    let detector = ArucoDetector::new(ArucoDict::Dict6X6_250);
    let markers = detector.detect_markers(&frame)?;

    println!("Detected {} ArUco marker(s):", markers.len());
    for m in &markers {
        println!(" - Marker ID: {}", m.id);
        println!(" - Corner points: {:?}", m.corners);
    }

    // 3. Estimate single marker poses
    let camera_matrix = [[500.0, 0.0, 200.0], [0.0, 500.0, 200.0], [0.0, 0.0, 1.0]];
    let dist_coeffs = vec![0.0; 5];

    let (rvecs, tvecs) =
        detector.estimate_pose_single_markers(&markers, 0.05, &camera_matrix, &dist_coeffs)?;
    for i in 0..markers.len() {
        println!(
            "Marker {} translation vector (tvec): {:?}, rotation vector (rvec): {:?}",
            markers[i].id, tvecs[i], rvecs[i]
        );
    }

    Ok(())
}
