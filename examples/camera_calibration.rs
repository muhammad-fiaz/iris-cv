use iris::prelude::*;

fn main() -> Result<()> {
    println!("--- Camera Calibration & Geometry Example ---");

    // 1. Generate checkerboard coordinates for calibration
    let mut object_pts = Vec::new();
    let mut image_pts = Vec::new();

    // Imagine a single calibration frame with 4 point pairs
    object_pts.push(vec![
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.0),
        Point::new(1.0, 1.0),
        Point::new(0.0, 1.0),
    ]);

    image_pts.push(vec![
        Point::new(100.0, 100.0),
        Point::new(200.0, 105.0),
        Point::new(205.0, 205.0),
        Point::new(98.0, 200.0),
    ]);

    // 2. Calibrate camera
    let (camera_matrix, dist_coeffs) =
        CameraCalibration::calibrate_camera(&object_pts, &image_pts, Size::new(640, 480))?;

    println!("Calibrated Camera Matrix:");
    for row in &camera_matrix {
        println!("  {:?}", row);
    }
    println!("Distortion Coefficients: {:?}", dist_coeffs);

    // 3. Project 3D points
    let rvec = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let tvec = [[0.0, 0.0, 5.0]];

    let test_points = vec![Point::new(0.5, 0.5)];
    let projected = CameraCalibration::project_points(
        &test_points,
        &rvec,
        &tvec,
        &camera_matrix,
        &dist_coeffs,
    )?;

    println!(
        "Projected 3D point (0.5, 0.5, 5.0) -> pixel: {:?}",
        projected.first()
    );

    // 4. Find Homography matrix
    let h_mat = CameraCalibration::find_homography(&object_pts[0], &image_pts[0])?;
    println!("Estimated 3x3 Homography Matrix:");
    for row in &h_mat {
        println!("  {:?}", row);
    }

    Ok(())
}
