use crate::error::Result;
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

impl<B: Backend> Image<B> {
    /// Applies a bilateral filter to smooth the image while preserving sharp edges.
    pub fn bilateral_filter(&self, d: isize, sigma_color: f64, sigma_space: f64) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        let rad = if d <= 0 {
            // Estimate diameter from sigma space
            (sigma_space * 3.0).round() as isize
        } else {
            d / 2
        };

        let space_coeff = -1.0 / (2.0 * sigma_space * sigma_space);
        let color_coeff = -1.0 / (2.0 * sigma_color * sigma_color);

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out_vals
                .par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(idx, row)| {
                    let ch = idx / h;
                    let y = idx % h;

                    for x in 0..w {
                        let mut sum_vals = 0.0f64;
                        let mut sum_weights = 0.0f64;
                        let center_val = f64::from(flat_vals[ch * h * w + y * w + x]);

                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            if py >= 0 && py < h as isize {
                                for kx in -rad..=rad {
                                    let px = x as isize + kx;
                                    if px >= 0 && px < w as isize {
                                        let neighbor_val = f64::from(
                                            flat_vals
                                                [ch * h * w + (py as usize) * w + (px as usize)],
                                        );

                                        // Space distance
                                        let r2 = (kx * kx + ky * ky) as f64;
                                        // Color difference
                                        let diff = neighbor_val - center_val;
                                        let diff2 = diff * diff;

                                        let space_weight = (r2 * space_coeff).exp();
                                        let color_weight = (diff2 * color_coeff).exp();
                                        let weight = space_weight * color_weight;

                                        sum_vals += neighbor_val * weight;
                                        sum_weights += weight;
                                    }
                                }
                            }
                        }

                        if sum_weights > 0.0 {
                            row[x] = (sum_vals / sum_weights) as f32;
                        } else {
                            row[x] = center_val as f32;
                        }
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let mut sum_vals = 0.0f64;
                        let mut sum_weights = 0.0f64;
                        let center_val = flat_vals[ch * h * w + y * w + x] as f64;

                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            if py >= 0 && py < h as isize {
                                for kx in -rad..=rad {
                                    let px = x as isize + kx;
                                    if px >= 0 && px < w as isize {
                                        let neighbor_val = flat_vals
                                            [ch * h * w + (py as usize) * w + (px as usize)]
                                            as f64;

                                        // Space distance
                                        let r2 = (kx * kx + ky * ky) as f64;
                                        // Color difference
                                        let diff = neighbor_val - center_val;
                                        let diff2 = diff * diff;

                                        let space_weight = (r2 * space_coeff).exp();
                                        let color_weight = (diff2 * color_coeff).exp();
                                        let weight = space_weight * color_weight;

                                        sum_vals += neighbor_val * weight;
                                        sum_weights += weight;
                                    }
                                }
                            }
                        }

                        if sum_weights > 0.0 {
                            out_vals[ch * h * w + y * w + x] = (sum_vals / sum_weights) as f32;
                        } else {
                            out_vals[ch * h * w + y * w + x] = center_val as f32;
                        }
                    }
                }
            }
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Performs separable 2D convolution: first filtering horizontally with `kernel_x`,
    /// then filtering vertically with `kernel_y`.
    pub fn sep_filter_2d(&self, kernel_x: &[f32], kernel_y: &[f32]) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        // Step 1: Filter Horizontally
        let mut temp_vals = vec![0.0f32; c * h * w];
        let rad_x = (kernel_x.len() / 2) as isize;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            temp_vals
                .par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(idx, row)| {
                    let ch = idx / h;
                    let y = idx % h;
                    for x in 0..w {
                        let mut sum = 0.0f64;
                        for kx in -rad_x..=rad_x {
                            let px = x as isize + kx;
                            let px_clamped = px.clamp(0, w as isize - 1) as usize;
                            let weight = f64::from(kernel_x[(kx + rad_x) as usize]);
                            sum += f64::from(flat_vals[ch * h * w + y * w + px_clamped]) * weight;
                        }
                        row[x] = sum as f32;
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let mut sum = 0.0f64;
                        for kx in -rad_x..=rad_x {
                            let px = x as isize + kx;
                            let px_clamped = px.clamp(0, w as isize - 1) as usize;
                            let weight = kernel_x[(kx + rad_x) as usize] as f64;
                            sum += flat_vals[ch * h * w + y * w + px_clamped] as f64 * weight;
                        }
                        temp_vals[ch * h * w + y * w + x] = sum as f32;
                    }
                }
            }
        }

        // Step 2: Filter Vertically
        let mut out_vals = vec![0.0f32; c * h * w];
        let rad_y = (kernel_y.len() / 2) as isize;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out_vals
                .par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(idx, row)| {
                    let ch = idx / h;
                    let y = idx % h;
                    for x in 0..w {
                        let mut sum = 0.0f64;
                        for ky in -rad_y..=rad_y {
                            let py = y as isize + ky;
                            let py_clamped = py.clamp(0, h as isize - 1) as usize;
                            let weight = f64::from(kernel_y[(ky + rad_y) as usize]);
                            sum += f64::from(temp_vals[ch * h * w + py_clamped * w + x]) * weight;
                        }
                        row[x] = sum as f32;
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let mut sum = 0.0f64;
                        for ky in -rad_y..=rad_y {
                            let py = y as isize + ky;
                            let py_clamped = py.clamp(0, h as isize - 1) as usize;
                            let weight = kernel_y[(ky + rad_y) as usize] as f64;
                            sum += temp_vals[ch * h * w + py_clamped * w + x] as f64 * weight;
                        }
                        out_vals[ch * h * w + y * w + x] = sum as f32;
                    }
                }
            }
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{test_device, TestBackend};

    #[test]
    fn test_bilateral_and_separable() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor_data = TensorData::new(flat_data, [3, 8, 8]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(tensor_data, &device));

        let bilateral = img.bilateral_filter(3, 0.1, 1.0).unwrap();
        assert_eq!(bilateral.shape(), [3, 8, 8]);

        let kernel_x = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let kernel_y = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let sep = img.sep_filter_2d(&kernel_x, &kernel_y).unwrap();
        assert_eq!(sep.shape(), [3, 8, 8]);
    }
}
