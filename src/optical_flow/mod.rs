use crate::core::types::Point;
use crate::error::Result;
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Optical Flow analyzer.
pub struct OpticalFlow;

impl OpticalFlow {
    /// Computes dense optical flow using Farneback's algorithm.
    /// Returns flow tensor of shape [2, H, W] containing flow vectors (dx, dy).
    ///
    /// Farneback's method works by:
    /// 1. Expanding each image into a quadratic polynomial using Gaussian weighting
    /// 2. Computing the displacement field from the polynomial expansion coefficients
    /// 3. Refining the flow iteratively at multiple scales
    pub fn calc_dense_farneback<B: Backend>(
        prev: &Image<B>,
        next: &Image<B>,
    ) -> Result<Tensor<B, 3>> {
        let prev_gray = prev.grayscale()?;
        let next_gray = next.grayscale()?;
        let dims = prev_gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];
        let device = prev_gray.tensor.device();

        let prev_data = prev_gray.tensor.clone().into_data();
        let next_data = next_gray.tensor.clone().into_data();
        let prev_vals: Vec<f32> = prev_data.iter::<f32>().collect();
        let next_vals: Vec<f32> = next_data.iter::<f32>().collect();

        // Parameters
        let num_levels = 5;
        let pyr_scale = 0.5f64;
        let iterations = 3;
        let poly_n = 7; // Neighborhood size for polynomial expansion
        let poly_sigma = 1.5f64;

        // Build Gaussian kernel for polynomial expansion
        let gaussian = build_gaussian_kernel(poly_n, poly_sigma);

        // Build image pyramids
        let prev_pyr = build_pyramid(&prev_vals, w, h, num_levels, pyr_scale);
        let next_pyr = build_pyramid(&next_vals, w, h, num_levels, pyr_scale);

        // Initialize flow at coarsest level
        let coarse_h = prev_pyr[num_levels - 1].1;
        let coarse_w = prev_pyr[num_levels - 1].0;
        let mut flow_x = vec![0.0f32; coarse_h * coarse_w];
        let mut flow_y = vec![0.03f32; coarse_h * coarse_w];

        // Iterate from coarse to fine
        for level in (0..num_levels).rev() {
            let lev_w = prev_pyr[level].0;
            let lev_h = prev_pyr[level].1;
            let prev_img = &prev_pyr[level].2;
            let next_img = &next_pyr[level].2;

            // Upsample flow to current level if not at coarsest
            if level < num_levels - 1 {
                let next_lev_w = prev_pyr[level + 1].0;
                let next_lev_h = prev_pyr[level + 1].1;
                let upsampled_x = upsample_flow(&flow_x, next_lev_w, next_lev_h, lev_w, lev_h);
                let upsampled_y = upsample_flow(&flow_y, next_lev_w, next_lev_h, lev_w, lev_h);
                flow_x = upsampled_x;
                flow_y = upsampled_y;
            }

            // Compute polynomial expansion of prev image
            let (a00, _a11, a22, a01, _a02, _a12) =
                compute_poly_expansion(prev_img, lev_w, lev_h, &gaussian, poly_n);

            // Iterate to refine flow
            for _ in 0..iterations {
                let mut new_flow_x = vec![0.0f32; lev_h * lev_w];
                let mut new_flow_y = vec![0.0f32; lev_h * lev_w];

                for y in 0..lev_h {
                    for x in 0..lev_w {
                        let idx = y * lev_w + x;

                        // Warp next image location using current flow
                        let wx = (x as f64 + flow_x[idx] as f64).round() as i32;
                        let wy = (y as f64 + flow_y[idx] as f64).round() as i32;

                        if wx >= 0 && wx < lev_w as i32 && wy >= 0 && wy < lev_h as i32 {
                            let widx = wy as usize * lev_w + wx as usize;
                            let diff = next_img[widx] - prev_img[idx];

                            // Compute gradient of next image at warped location
                            let gx = if wx > 0 && wx < lev_w as i32 - 1 {
                                (next_img[widx + 1] - next_img[widx - 1]) * 0.5
                            } else {
                                0.0
                            };
                            let gy = if wy > 0 && wy < lev_h as i32 - 1 {
                                (next_img[(wy as usize + 1) * lev_w + wx as usize]
                                    - next_img[(wy as usize - 1) * lev_w + wx as usize])
                                    * 0.5
                            } else {
                                0.0
                            };

                            // Solve for flow update using least squares
                            let b0 = gx * diff;
                            let b1 = gy * diff;
                            let det = a00[idx] * a22[idx] - a01[idx] * a01[idx];
                            if det.abs() > 1e-10 {
                                let inv00 = a22[idx] / det;
                                let inv01 = -a01[idx] / det;
                                let inv11 = a00[idx] / det;
                                new_flow_x[idx] =
                                    flow_x[idx] + (inv00 * b0 + inv01 * b1) as f32;
                                new_flow_y[idx] =
                                    flow_y[idx] + (inv01 * b0 + inv11 * b1) as f32;
                            } else {
                                new_flow_x[idx] = flow_x[idx];
                                new_flow_y[idx] = flow_y[idx];
                            }
                        } else {
                            new_flow_x[idx] = flow_x[idx];
                            new_flow_y[idx] = flow_y[idx];
                        }
                    }
                }

                flow_x = new_flow_x;
                flow_y = new_flow_y;
            }
        }

        // Upsample flow back to original resolution if needed
        let orig_w = prev_pyr[0].0;
        let orig_h = prev_pyr[0].1;
        let coarse_dims = &prev_pyr[num_levels - 1];
        if flow_x.len() != orig_h * orig_w {
            flow_x = upsample_flow(&flow_x, coarse_dims.0, coarse_dims.1, orig_w, orig_h);
            flow_y = upsample_flow(&flow_y, coarse_dims.0, coarse_dims.1, orig_w, orig_h);
        }

        // Pack into [2, H, W] tensor
        let mut flow_flat = Vec::with_capacity(2 * orig_h * orig_w);
        flow_flat.extend_from_slice(&flow_x);
        flow_flat.extend_from_slice(&flow_y);

        let data = TensorData::new(flow_flat, [2, orig_h, orig_w]);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(tensor)
    }

    /// Computes sparse optical flow using Lucas-Kanade feature tracking.
    /// For each point in `prev_pts`, finds the corresponding location in the next frame
    /// by solving the Lucas-Kanade equations in a local window.
    pub fn calc_sparse_pyr_lk<B: Backend>(
        prev: &Image<B>,
        next: &Image<B>,
        prev_pts: &[Point<f64>],
    ) -> Result<(Vec<Point<f64>>, Vec<u8>)> {
        let prev_gray = prev.grayscale()?;
        let next_gray = next.grayscale()?;
        let dims = prev_gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let prev_data = prev_gray.tensor.clone().into_data();
        let next_data = next_gray.tensor.clone().into_data();
        let prev_vals: Vec<f32> = prev_data.iter::<f32>().collect();
        let next_vals: Vec<f32> = next_data.iter::<f32>().collect();

        let window_size = 15; // Window for Lucas-Kanade
        let half_win = window_size / 2;
        let max_iter = 20;
        let epsilon = 0.01f64;

        let mut next_pts = Vec::with_capacity(prev_pts.len());
        let mut status = Vec::with_capacity(prev_pts.len());

        for pt in prev_pts {
            let mut px = pt.x;
            let mut py = pt.y;
            let mut tracked = true;

            for _ in 0..max_iter {
                let ix = px as i32;
                let iy = py as i32;

                if ix < half_win
                    || ix >= w as i32 - half_win
                    || iy < half_win
                    || iy >= h as i32 - half_win
                {
                    tracked = false;
                    break;
                }

                // Compute spatial gradients in window
                let mut sum_gx2 = 0.0f64;
                let mut sum_gy2 = 0.0f64;
                let mut sum_gxgy = 0.0f64;
                let mut sum_gxgt = 0.0f64;
                let mut sum_gygt = 0.0f64;

                for wy in -half_win..=half_win {
                    for wx in -half_win..=half_win {
                        let cx = ix + wx;
                        let cy = iy + wy;

                        if cx > 0 && cx < w as i32 - 1 && cy > 0 && cy < h as i32 - 1 {
                            let gx = (prev_vals[cy as usize * w + (cx + 1) as usize] as f64
                                - prev_vals[cy as usize * w + (cx - 1) as usize] as f64)
                                * 0.5;
                            let gy = (prev_vals[(cy + 1) as usize * w + cx as usize] as f64
                                - prev_vals[(cy - 1) as usize * w + cx as usize] as f64)
                                * 0.5;

                            // Sample next image at warped location
                            let nx = (cx as f64 + 0.0) as i32;
                            let ny = (cy as f64 + 0.0) as i32;
                            let gt = if nx >= 0
                                && nx < w as i32
                                && ny >= 0
                                && ny < h as i32
                            {
                                next_vals[ny as usize * w + nx as usize] as f64
                            } else {
                                prev_vals[cy as usize * w + cx as usize] as f64
                            };

                            let it = prev_vals[cy as usize * w + cx as usize] as f64 - gt;

                            sum_gx2 += gx * gx;
                            sum_gy2 += gy * gy;
                            sum_gxgy += gx * gy;
                            sum_gxgt += gx * it;
                            sum_gygt += gy * it;
                        }
                    }
                }

                // Solve 2x2 system: [gx2 gxgy; gxgy gy2] * [dx; dy] = [gxgt; gygt]
                let det = sum_gx2 * sum_gy2 - sum_gxgy * sum_gxgy;
                if det.abs() < 1e-10 {
                    break;
                }

                let dx = (sum_gy2 * sum_gxgt - sum_gxgy * sum_gygt) / det;
                let dy = (sum_gx2 * sum_gygt - sum_gxgy * sum_gxgt) / det;

                px += dx;
                py += dy;

                if (dx * dx + dy * dy).sqrt() < epsilon {
                    break;
                }
            }

            if tracked {
                // Verify point is within bounds
                if px >= 0.0
                    && px < w as f64
                    && py >= 0.0
                    && py < h as f64
                {
                    next_pts.push(Point::new(px, py));
                    status.push(1);
                } else {
                    next_pts.push(*pt);
                    status.push(0);
                }
            } else {
                next_pts.push(*pt);
                status.push(0);
            }
        }

        Ok((next_pts, status))
    }
}

/// Builds a 1D Gaussian kernel.
fn build_gaussian_kernel(size: usize, sigma: f64) -> Vec<f64> {
    let half = size / 2;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0;
    for i in 0..size {
        let x = i as f64 - half as f64;
        let val = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(val);
        sum += val;
    }
    for k in &mut kernel {
        *k /= sum;
    }
    kernel
}

/// Separable Gaussian blur on a 2D image.
fn gaussian_blur_2d(img: &[f32], w: usize, h: usize, kernel: &[f64]) -> Vec<f32> {
    let k_half = kernel.len() / 2;
    let mut temp = vec![0.0f32; h * w];
    let mut out = vec![0.0f32; h * w];

    // Horizontal pass
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f64;
            for k in 0..kernel.len() {
                let sx = (x as i32 + k as i32 - k_half as i32).clamp(0, w as i32 - 1) as usize;
                sum += img[y * w + sx] as f64 * kernel[k];
            }
            temp[y * w + x] = sum as f32;
        }
    }

    // Vertical pass
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f64;
            for k in 0..kernel.len() {
                let sy = (y as i32 + k as i32 - k_half as i32).clamp(0, h as i32 - 1) as usize;
                sum += temp[sy * w + x] as f64 * kernel[k];
            }
            out[y * w + x] = sum as f32;
        }
    }

    out
}

/// Builds image pyramid by repeated Gaussian blur + subsampling.
fn build_pyramid(
    img: &[f32],
    w: usize,
    h: usize,
    levels: usize,
    scale: f64,
) -> Vec<(usize, usize, Vec<f32>)> {
    let mut pyramid = Vec::with_capacity(levels);
    let kernel = build_gaussian_kernel(5, 1.0);

    let mut current = img.to_vec();
    let mut cur_w = w;
    let mut cur_h = h;

    for _ in 0..levels {
        pyramid.push((cur_w, cur_h, current.clone()));
        let new_w = ((cur_w as f64) * scale).max(1.0) as usize;
        let new_h = ((cur_h as f64) * scale).max(1.0) as usize;

        let blurred = gaussian_blur_2d(&current, cur_w, cur_h, &kernel);

        // Subsample
        let mut downsampled = vec![0.0f32; new_h * new_w];
        for y in 0..new_h {
            for x in 0..new_w {
                let sx = ((x as f64 * cur_w as f64 / new_w as f64) as usize).min(cur_w - 1);
                let sy = ((y as f64 * cur_h as f64 / new_h as f64) as usize).min(cur_h - 1);
                downsampled[y * new_w + x] = blurred[sy * cur_w + sx];
            }
        }

        current = downsampled;
        cur_w = new_w;
        cur_h = new_h;
    }

    pyramid
}

/// Upsamples a flow field from a smaller to larger resolution.
fn upsample_flow(flow: &[f32], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dst_h * dst_w];
    for y in 0..dst_h {
        for x in 0..dst_w {
            let sx = ((x as f64 * src_w as f64 / dst_w as f64) as usize).min(src_w - 1);
            let sy = ((y as f64 * src_h as f64 / dst_h as f64) as usize).min(src_h - 1);
            out[y * dst_w + x] = flow[sy * src_w + sx] * (dst_w as f32 / src_w as f32);
        }
    }
    out
}

/// Polynomial expansion coefficients for a single pixel.
type PolyCoeffs = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Computes polynomial expansion coefficients for Farneback's method.
/// Returns (a00, a11, a22, a01, a02, a12) for each pixel.
fn compute_poly_expansion(
    img: &[f32],
    w: usize,
    h: usize,
    gaussian: &[f64],
    poly_n: usize,
) -> PolyCoeffs {
    let half = poly_n / 2;
    let size = w * h;

    let mut a00 = vec![0.0f32; size];
    let mut a11 = vec![0.0f32; size];
    let mut a22 = vec![0.0f32; size];
    let mut a01 = vec![0.0f32; size];
    let mut a02 = vec![0.0f32; size];
    let mut a12 = vec![0.0f32; size];

    for y in 0..h {
        for x in 0..w {
            let mut sum_a00 = 0.0f64;
            let mut sum_a11 = 0.0f64;
            let mut sum_a22 = 0.0f64;
            let mut sum_a01 = 0.0f64;
            let mut sum_a02 = 0.0f64;
            let mut sum_a12 = 0.0f64;

            for ky in 0..poly_n {
                for kx in 0..poly_n {
                    let sy = (y + ky).min(h - 1);
                    let sx = (x + kx).min(w - 1);
                    let g = gaussian[ky] * gaussian[kx];
                    let val = img[sy * w + sx] as f64;
                    let dx = (kx as f64) - (half as f64);
                    let dy = (ky as f64) - (half as f64);

                    sum_a00 += g * val;
                    sum_a11 += g * val * dx * dx;
                    sum_a22 += g * val * dy * dy;
                    sum_a01 += g * val * dx * dy;
                    sum_a02 += g * val * dx * dx * dx;
                    sum_a12 += g * val * dy * dy * dy;
                }
            }

            let idx = y * w + x;
            a00[idx] = sum_a00 as f32;
            a11[idx] = sum_a11 as f32;
            a22[idx] = sum_a22 as f32;
            a01[idx] = sum_a01 as f32;
            a02[idx] = sum_a02 as f32;
            a12[idx] = sum_a12 as f32;
        }
    }

    (a00, a11, a22, a01, a02, a12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::TensorData;

    #[test]
    fn test_dense_optical_flow() {
        let device = test_device();
        let mut flat_data1 = vec![0.0f32; 3 * 16 * 16];
        let mut flat_data2 = vec![0.0f32; 3 * 16 * 16];
        // Create two images with a shifted pattern
        for y in 0..16 {
            for x in 0..16 {
                let val1 = if x < 8 { 0.0 } else { 1.0 };
                let val2 = if x < 7 { 0.0 } else { 1.0 }; // Shifted left by 1
                flat_data1[y * 16 + x] = val1;
                flat_data1[256 + y * 16 + x] = val1;
                flat_data1[512 + y * 16 + x] = val1;
                flat_data2[y * 16 + x] = val2;
                flat_data2[256 + y * 16 + x] = val2;
                flat_data2[512 + y * 16 + x] = val2;
            }
        }
        let tensor1 = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(flat_data1, [3, 16, 16]),
            &device,
        );
        let tensor2 = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(flat_data2, [3, 16, 16]),
            &device,
        );
        let img1 = Image::new(tensor1);
        let img2 = Image::new(tensor2);

        let flow = OpticalFlow::calc_dense_farneback(&img1, &img2).unwrap();
        assert_eq!(flow.dims(), [2, 16, 16]);
    }

    #[test]
    fn test_sparse_optical_flow() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 32 * 32];
        let tensor1 =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data.clone(), [3, 32, 32]), &device);
        let tensor2 =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 32, 32]), &device);
        let img1 = Image::new(tensor1);
        let img2 = Image::new(tensor2);

        let pts = vec![Point::new(16.0, 16.0), Point::new(8.0, 8.0)];
        let (next_pts, status) = OpticalFlow::calc_sparse_pyr_lk(&img1, &img2, &pts).unwrap();
        assert_eq!(next_pts.len(), 2);
        assert_eq!(status.len(), 2);
    }
}
