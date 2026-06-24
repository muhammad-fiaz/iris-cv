pub mod bilateral;

use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

impl<B: Backend> Image<B> {
    /// Applies a box filter to blur the image with the specified kernel size.
    pub fn box_blur(self, kernel_size: usize) -> Result<Self> {
        if kernel_size.is_multiple_of(2) {
            return Err(IrisError::InvalidParameter(
                "Kernel size must be odd".into(),
            ));
        }

        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        let rad = (kernel_size / 2) as isize;

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
                        let mut sum = 0.0f32;
                        let mut count = 0.0f32;

                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            if py >= 0 && py < h as isize {
                                for kx in -rad..=rad {
                                    let px = x as isize + kx;
                                    if px >= 0 && px < w as isize {
                                        sum += flat_vals
                                            [ch * h * w + (py as usize) * w + (px as usize)];
                                        count += 1.0;
                                    }
                                }
                            }
                        }
                        row[x] = sum / count;
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let mut sum = 0.0f32;
                        let mut count = 0.0f32;

                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            if py >= 0 && py < h as isize {
                                for kx in -rad..=rad {
                                    let px = x as isize + kx;
                                    if px >= 0 && px < w as isize {
                                        sum += flat_vals
                                            [ch * h * w + (py as usize) * w + (px as usize)];
                                        count += 1.0;
                                    }
                                }
                            }
                        }
                        out_vals[ch * h * w + y * w + x] = sum / count;
                    }
                }
            }
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Applies a Gaussian blur filter to the image.
    pub fn gaussian_blur(self, kernel_size: usize, sigma: f64) -> Result<Self> {
        if kernel_size.is_multiple_of(2) {
            return Err(IrisError::InvalidParameter(
                "Kernel size must be odd".into(),
            ));
        }

        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        let rad = (kernel_size / 2) as isize;

        // Generate Gaussian kernel
        let mut kernel = vec![vec![0.0f64; kernel_size]; kernel_size];
        let mut sum = 0.0f64;

        let s2 = 2.0 * sigma * sigma;

        for ky in -rad..=rad {
            for kx in -rad..=rad {
                let r = (kx * kx + ky * ky) as f64;
                let val = (-r / s2).exp();
                kernel[(ky + rad) as usize][(kx + rad) as usize] = val;
                sum += val;
            }
        }

        // Normalize kernel
        for row in &mut kernel {
            for val in row {
                *val /= sum;
            }
        }

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
                        let mut blur_sum = 0.0f64;
                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            let py_clamped = py.clamp(0, h as isize - 1) as usize;
                            for kx in -rad..=rad {
                                let px = x as isize + kx;
                                let px_clamped = px.clamp(0, w as isize - 1) as usize;
                                let weight = kernel[(ky + rad) as usize][(kx + rad) as usize];
                                blur_sum += f64::from(flat_vals[ch * h * w + py_clamped * w + px_clamped])
                                    * weight;
                            }
                        }
                        row[x] = blur_sum as f32;
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let mut blur_sum = 0.0f64;
                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            let py_clamped = py.clamp(0, h as isize - 1) as usize;
                            for kx in -rad..=rad {
                                let px = x as isize + kx;
                                let px_clamped = px.clamp(0, w as isize - 1) as usize;
                                let weight = kernel[(ky + rad) as usize][(kx + rad) as usize];
                                blur_sum += flat_vals[ch * h * w + py_clamped * w + px_clamped]
                                    as f64
                                    * weight;
                            }
                        }
                        out_vals[ch * h * w + y * w + x] = blur_sum as f32;
                    }
                }
            }
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Applies a median filter to reduce salt-and-pepper noise.
    pub fn median_blur(self, kernel_size: usize) -> Result<Self> {
        if kernel_size.is_multiple_of(2) {
            return Err(IrisError::InvalidParameter(
                "Kernel size must be odd".into(),
            ));
        }

        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        let rad = (kernel_size / 2) as isize;

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
                        let mut neighbors = Vec::with_capacity(kernel_size * kernel_size);

                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            if py >= 0 && py < h as isize {
                                for kx in -rad..=rad {
                                    let px = x as isize + kx;
                                    if px >= 0 && px < w as isize {
                                        neighbors.push(
                                            flat_vals
                                                [ch * h * w + (py as usize) * w + (px as usize)],
                                        );
                                    }
                                }
                            }
                        }
                        neighbors
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let median = neighbors[neighbors.len() / 2];
                        row[x] = median;
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let mut neighbors = Vec::with_capacity(kernel_size * kernel_size);

                        for ky in -rad..=rad {
                            let py = y as isize + ky;
                            if py >= 0 && py < h as isize {
                                for kx in -rad..=rad {
                                    let px = x as isize + kx;
                                    if px >= 0 && px < w as isize {
                                        neighbors.push(
                                            flat_vals
                                                [ch * h * w + (py as usize) * w + (px as usize)],
                                        );
                                    }
                                }
                            }
                        }
                        neighbors
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let median = neighbors[neighbors.len() / 2];
                        out_vals[ch * h * w + y * w + x] = median;
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
    use burn::backend::wgpu::Wgpu;

    #[test]
    fn test_filters_blur() {
        let device = Default::default();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor_data = TensorData::new(flat_data, [3, 8, 8]);
        let img = Image::new(Tensor::<Wgpu, 3>::from_data(tensor_data, &device));

        let boxed = img.clone().box_blur(3).unwrap();
        assert_eq!(boxed.shape(), [3, 8, 8]);

        let gauss = img.clone().gaussian_blur(3, 1.0).unwrap();
        assert_eq!(gauss.shape(), [3, 8, 8]);

        let median = img.clone().median_blur(3).unwrap();
        assert_eq!(median.shape(), [3, 8, 8]);
    }
}
