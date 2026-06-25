use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Different types of thresholding operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThresholdType {
    Binary,
    BinaryInv,
    Trunc,
    ToZero,
    ToZeroInv,
}

/// Strategy for adaptive threshold computation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AdaptiveMethod {
    /// Mean of the blockSize x blockSize neighborhood.
    MeanC,
    /// Gaussian-weighted sum of the blockSize x blockSize neighborhood.
    GaussianC,
}

impl<B: Backend> Image<B> {
    /// Applies a fixed-level threshold to each array element.
    pub fn threshold(&self, thresh: f32, maxval: f32, thresh_type: ThresholdType) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        {
            use rayon::prelude::*;
            out_vals
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out_val)| {
                    let val = flat_vals[i];
                    *out_val = match thresh_type {
                        ThresholdType::Binary => {
                            if val > thresh { maxval } else { 0.0 }
                        }
                        ThresholdType::BinaryInv => {
                            if val > thresh { 0.0 } else { maxval }
                        }
                        ThresholdType::Trunc => {
                            if val > thresh { thresh } else { val }
                        }
                        ThresholdType::ToZero => {
                            if val > thresh { val } else { 0.0 }
                        }
                        ThresholdType::ToZeroInv => {
                            if val > thresh { 0.0 } else { val }
                        }
                    };
                });
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Automatically computes a threshold using Otsu's method and applies binary thresholding.
    pub fn threshold_otsu(&self, maxval: f32) -> Result<Self> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut hist = [0u32; 256];
        for &val in &flat_vals {
            let bin = (val.clamp(0.0, 1.0) * 255.0) as usize;
            hist[bin] += 1;
        }

        let total = (h * w) as f32;
        let mut sum = 0.0f32;
        for (i, &count) in hist.iter().enumerate() {
            sum += (i as f32) * (count as f32);
        }

        let mut sum_b = 0.0f32;
        let mut w_b = 0.0f32;
        let mut max_var = 0.0f32;
        let mut threshold = 0;

        for (t, &count) in hist.iter().enumerate() {
            w_b += count as f32;
            if w_b == 0.0 {
                continue;
            }
            let w_f = total - w_b;
            if w_f == 0.0 {
                break;
            }

            sum_b += (t as f32) * (count as f32);
            let m_b = sum_b / w_b;
            let m_f = (sum - sum_b) / w_f;

            let var_between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
            if var_between > max_var {
                max_var = var_between;
                threshold = t;
            }
        }

        let thresh_float = (threshold as f32) / 255.0;
        self.threshold(thresh_float, maxval, ThresholdType::Binary)
    }

    /// Automatically computes a threshold using the Triangle method and applies binary thresholding.
    ///
    /// The Triangle method draws a line from the histogram peak to the far end of the histogram,
    /// then finds the threshold at the point of maximum distance from the line.
    pub fn threshold_triangle(&self, maxval: f32) -> Result<Self> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let _h = dims[1];
        let _w = dims[2];

        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut hist = [0u32; 256];
        for &val in &flat_vals {
            let bin = (val.clamp(0.0, 1.0) * 255.0) as usize;
            hist[bin] += 1;
        }

        // Find the peak of the histogram
        let mut peak = 0;
        let mut max_count = 0u32;
        for (i, &count) in hist.iter().enumerate() {
            if count > max_count {
                max_count = count;
                peak = i;
            }
        }

        // Find the far end (last non-zero bin)
        let mut last = 255;
        for i in (0..256).rev() {
            if hist[i] > 0 {
                last = i;
                break;
            }
        }

        if peak == last {
            let thresh_float = (peak as f32) / 255.0;
            return self.threshold(thresh_float, maxval, ThresholdType::Binary);
        }

        // Draw line from (peak, max_count) to (last, 0)
        let dx = (last as f64) - (peak as f64);
        let dy = 0.0 - (max_count as f64);
        let line_len = (dx * dx + dy * dy).sqrt();

        // For each bin between peak and last, compute distance from the line
        let mut max_dist = 0.0f64;
        let mut threshold = peak;

        for i in peak..=last {
            let px = i as f64;
            let py = hist[i] as f64;
            // Distance from point (px, py) to line from (peak, max_count) to (last, 0)
            let dist = ((dy * px - dx * py + (last as f64) * (max_count as f64)
                - (peak as f64) * 0.0)
                / line_len)
                .abs();
            if dist > max_dist {
                max_dist = dist;
                threshold = i;
            }
        }

        let thresh_float = (threshold as f32) / 255.0;
        self.threshold(thresh_float, maxval, ThresholdType::Binary)
    }

    /// Applies adaptive thresholding where each pixel gets its own threshold
    /// based on a neighborhood statistic.
    pub fn adaptive_threshold(
        &self,
        maxval: f32,
        method: AdaptiveMethod,
        block_size: usize,
        c: f32,
    ) -> Result<Self> {
        if block_size == 0 || block_size.is_multiple_of(2) {
            return Err(IrisError::InvalidParameter(
                "block_size must be a positive odd number".into(),
            ));
        }

        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let tensor_data = gray.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; h * w];

        let half = block_size / 2;

        // Build integral image for fast neighborhood sum computation
        let mut integral = vec![0.0f64; (h + 1) * (w + 1)];
        for y in 0..h {
            let mut row_sum = 0.0f64;
            for x in 0..w {
                row_sum += flat_vals[y * w + x] as f64;
                integral[(y + 1) * (w + 1) + (x + 1)] =
                    integral[y * (w + 1) + (x + 1)] + row_sum;
            }
        }

        // Compute Gaussian weights for GaussianC method
        let gaussian_kernel = if method == AdaptiveMethod::GaussianC {
            let sigma = (block_size as f64) / 6.0;
            let mut kernel = Vec::with_capacity(block_size * block_size);
            for ky in 0..block_size {
                for kx in 0..block_size {
                    let dy = (ky as f64) - (half as f64);
                    let dx = (kx as f64) - (half as f64);
                    let weight = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                    kernel.push(weight);
                }
            }
            let sum: f64 = kernel.iter().sum();
            for k in &mut kernel {
                *k /= sum;
            }
            Some(kernel)
        } else {
            None
        };

        let _total_pixels = block_size * block_size;

        for y in 0..h {
            for x in 0..w {
                let y1 = y.saturating_sub(half).min(h - 1);
                let y2 = (y + half).min(h - 1);
                let x1 = x.saturating_sub(half).min(w - 1);
                let x2 = (x + half).min(w - 1);

                let mean = if let Some(ref kernel) = gaussian_kernel {
                    // Gaussian-weighted sum
                    let mut weighted_sum = 0.0f64;
                    let mut ki = 0;
                    for ky in y1..=y2 {
                        for kx in x1..=x2 {
                            weighted_sum += flat_vals[ky * w + kx] as f64 * kernel[ki];
                            ki += 1;
                        }
                    }
                    weighted_sum
                } else {
                    // Mean: use integral image for O(1) neighborhood sum
                    let area = integral[(y2 + 1) * (w + 1) + (x2 + 1)]
                        - integral[y1 * (w + 1) + (x2 + 1)]
                        - integral[(y2 + 1) * (w + 1) + x1]
                        + integral[y1 * (w + 1) + x1];
                    let count = ((y2 - y1 + 1) * (x2 - x1 + 1)) as f64;
                    area / count
                };

                let pixel = flat_vals[y * w + x];
                if pixel > (mean as f32) - c {
                    out_vals[y * w + x] = maxval;
                } else {
                    out_vals[y * w + x] = 0.0;
                }
            }
        }

        let new_data = TensorData::new(out_vals, [1, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &gray.tensor.device());
        Ok(Image::new(new_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};

    #[test]
    fn test_threshold() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor_data = TensorData::new(flat_data, [3, 8, 8]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(tensor_data, &device));

        let thresh = img.threshold(0.4, 1.0, ThresholdType::Binary).unwrap();
        assert_eq!(thresh.shape(), [3, 8, 8]);

        let otsu = img.threshold_otsu(1.0).unwrap();
        assert_eq!(otsu.shape(), [3, 8, 8]);
    }

    #[test]
    fn test_triangle_threshold() {
        let device = test_device();
        let mut flat_data = vec![0.0f32; 16 * 16];
        // Create bimodal distribution
        for y in 0..16 {
            for x in 0..16 {
                if x < 8 {
                    flat_data[y * 16 + x] = 0.2;
                } else {
                    flat_data[y * 16 + x] = 0.8;
                }
            }
        }
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [1, 16, 16]), &device);
        let img = Image::new(tensor);
        let result = img.threshold_triangle(1.0).unwrap();
        assert_eq!(result.shape(), [1, 16, 16]);
    }

    #[test]
    fn test_adaptive_threshold() {
        let device = test_device();
        let mut flat_data = vec![0.0f32; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                flat_data[y * 16 + x] = (x as f32) / 16.0;
            }
        }
        let tensor = Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [1, 16, 16]), &device);
        let img = Image::new(tensor);

        let result = img.adaptive_threshold(1.0, AdaptiveMethod::MeanC, 3, 0.05).unwrap();
        assert_eq!(result.shape(), [1, 16, 16]);

        let result_gauss = img.adaptive_threshold(1.0, AdaptiveMethod::GaussianC, 5, 0.05).unwrap();
        assert_eq!(result_gauss.shape(), [1, 16, 16]);
    }
}
