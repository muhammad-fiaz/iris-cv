pub mod gradients;

use crate::error::Result;
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

impl<B: Backend> Image<B> {
    /// Applies Sobel edge detection to compute horizontal, vertical gradients, and magnitude.
    /// Returns the gradient magnitude image.
    pub fn sobel(&self) -> Result<Self> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let _c = dims[0]; // should be 1
        let h = dims[1];
        let w = dims[2];

        let device = gray.tensor.device();
        let tensor_data = gray.tensor.into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut mag_vals = vec![0.0f32; h * w];

        let kx = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
        let ky = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let mut gx = 0.0f32;
                let mut gy = 0.0f32;

                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let val =
                            flat_vals[(y as isize + dy) as usize * w + (x as isize + dx) as usize];
                        gx += val * kx[(dy + 1) as usize][(dx + 1) as usize] as f32;
                        gy += val * ky[(dy + 1) as usize][(dx + 1) as usize] as f32;
                    }
                }
                mag_vals[y * w + x] = (gx * gx + gy * gy).sqrt();
            }
        }

        let new_data = TensorData::new(mag_vals, [1, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Performs Canny edge detection.
    /// Steps: Grayscale -> Gaussian Blur -> Sobel Gradients -> Non-Maximum Suppression -> Hysteresis Thresholding.
    pub fn canny(&self, low_threshold: f32, high_threshold: f32) -> Result<Self> {
        // Step 1 & 2: Grayscale and Gaussian Blur (kernel_size = 5, sigma = 1.4)
        let blurred = self.grayscale()?.gaussian_blur(5, 1.4)?;

        let dims = blurred.tensor.dims();
        let h = dims[1];
        let w = dims[2];
        let device = blurred.tensor.device();

        let tensor_data = blurred.tensor.into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut gx_vals = vec![0.0f32; h * w];
        let mut gy_vals = vec![0.0f32; h * w];
        let mut mag_vals = vec![0.0f32; h * w];
        let mut angle_vals = vec![0.0f32; h * w];

        let kx = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
        let ky = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

        // Step 3: Compute Sobel gradients
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let mut gx = 0.0f32;
                let mut gy = 0.0f32;

                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let val =
                            flat_vals[(y as isize + dy) as usize * w + (x as isize + dx) as usize];
                        gx += val * kx[(dy + 1) as usize][(dx + 1) as usize] as f32;
                        gy += val * ky[(dy + 1) as usize][(dx + 1) as usize] as f32;
                    }
                }
                let idx = y * w + x;
                gx_vals[idx] = gx;
                gy_vals[idx] = gy;
                mag_vals[idx] = (gx * gx + gy * gy).sqrt();
                // Map angles to range [0, 180)
                let mut angle = gy.atan2(gx).to_degrees();
                if angle < 0.0 {
                    angle += 180.0;
                }
                angle_vals[idx] = angle;
            }
        }

        // Step 4: Non-Maximum Suppression (NMS)
        let mut nms_vals = vec![0.0f32; h * w];
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let idx = y * w + x;
                let angle = angle_vals[idx];
                let mag = mag_vals[idx];

                // Determine neighbors based on gradient direction
                let (mag1, mag2) =
                    if (0.0..22.5).contains(&angle) || (157.5..=180.0).contains(&angle) {
                        // Horizontal (0 degrees)
                        (mag_vals[y * w + (x + 1)], mag_vals[y * w + (x - 1)])
                    } else if (22.5..67.5).contains(&angle) {
                        // Diagonal (45 degrees)
                        (
                            mag_vals[(y - 1) * w + (x + 1)],
                            mag_vals[(y + 1) * w + (x - 1)],
                        )
                    } else if (67.5..112.5).contains(&angle) {
                        // Vertical (90 degrees)
                        (mag_vals[(y - 1) * w + x], mag_vals[(y + 1) * w + x])
                    } else {
                        // Diagonal (135 degrees)
                        (
                            mag_vals[(y - 1) * w + (x - 1)],
                            mag_vals[(y + 1) * w + (x + 1)],
                        )
                    };

                if mag >= mag1 && mag >= mag2 {
                    nms_vals[idx] = mag;
                } else {
                    nms_vals[idx] = 0.0;
                }
            }
        }

        // Step 5: Double Thresholding and Hysteresis
        let mut final_vals = vec![0.0f32; h * w];
        let mut strong_edges = Vec::new();

        // Label pixels: 0 = background, 1 = weak, 2 = strong
        let mut labels = vec![0u8; h * w];

        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let idx = y * w + x;
                let val = nms_vals[idx];

                if val >= high_threshold {
                    labels[idx] = 2;
                    strong_edges.push((x, y));
                    final_vals[idx] = 1.0; // Mark strong edge in final image
                } else if val >= low_threshold {
                    labels[idx] = 1;
                }
            }
        }

        // Hysteresis depth-first search (connected components)
        while let Some((x, y)) = strong_edges.pop() {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                        let nidx = (ny as usize) * w + (nx as usize);
                        if labels[nidx] == 1 {
                            // Upgrade weak edge to strong edge
                            labels[nidx] = 2;
                            final_vals[nidx] = 1.0;
                            strong_edges.push((nx as usize, ny as usize));
                        }
                    }
                }
            }
        }

        let new_data = TensorData::new(final_vals, [1, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }
}
