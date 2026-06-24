---
title: "Camera & Calibration"
description: "Capture video, calibrate camera matrices, estimate homography, solve PnP, and project 3D points with Iris."
keywords: ["camera", "calibration", "video capture", "homography", "camera matrix", "PnP", "projection"]
---

# Camera & Calibration

Interfacing with video sources and estimating camera matrices is essential for spatial mapping, stereo mapping, and 3D projection.

## Camera Interfacing

### Camera Capture
Queries frames from direct indices (integrated cameras) or file streams.

```rust
let device = Default::default();
let mut camera = Camera::<Backend>::open(CameraSource::Index(0), &device)?;

if camera.is_opened() {
    let frame = camera.capture()?;
    println!("Captured frame: {:?}", frame.shape());
}
```

### Video Capture & File Writing

```rust
// Load video file
let mut capture = VideoCapture::<Backend>::open("video.mp4", &device)?;
while let Some(frame) = capture.read()? {
    // Process frame
}

// Write video file
let mut writer = VideoWriter::<Backend>::create("output.mp4", 640, 480, 30.0)?;
writer.write(&processed_frame)?;
```

## Camera Calibration

Calibrates intrinsic matrices, distortion vectors, and projects coordinates.

### Camera Matrix Estimation

```rust
let object_points = vec![vec![Point::new(0.0, 0.0), Point::new(1.0, 0.0)]];
let image_points = vec![vec![Point::new(12.0, 10.0), Point::new(24.0, 10.0)]];
let size = Size::new(640, 480);

let (k, dist) = CameraCalibration::calibrate_camera(&object_points, &image_points, size)?;
```

### Projection of 3D Points

```rust
let pts_3d = vec![Point::new(0.5, 0.5)];
let rvec = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
let tvec = [[0.0, 0.0, 0.0]];

let projected_pts = CameraCalibration::project_points(&pts_3d, &rvec, &tvec, &k, &dist)?;
```

### Homography & Fundamental Matrices

```rust
// Estimate homography mapping
let h = CameraCalibration::find_homography(&src_pts, &dst_pts)?;

// Solve Perspective-n-Point pose estimation
let (rvec, tvec) = CameraCalibration::solve_pnp(&obj_pts, &img_pts, &k, &dist)?;
```
