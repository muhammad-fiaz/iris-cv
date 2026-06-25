use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Stereo block matcher using Sum of Absolute Differences (SAD).
pub struct StereoBlockMatcher {
    /// Size of the matching block (must be odd).
    block_size: i32,
    /// Maximum number of disparities to search.
    num_disparities: i32,
    /// Minimum disparity value.
    min_disparity: i32,
}

impl StereoBlockMatcher {
    /// Creates a new `StereoBlockMatcher`.
    ///
    /// * `block_size` - Side length of the square matching block (must be odd, >= 3).
    /// * `num_disparities` - Number of disparities to search (must be > 0, divisible by 16).
    pub fn new(block_size: i32, num_disparities: i32) -> Result<Self> {
        if block_size < 3 || block_size % 2 == 0 {
            return Err(IrisError::InvalidParameter(format!(
                "block_size must be odd and >= 3, got {block_size}"
            )));
        }
        if num_disparities <= 0 || num_disparities % 16 != 0 {
            return Err(IrisError::InvalidParameter(format!(
                "num_disparities must be > 0 and divisible by 16, got {num_disparities}"
            )));
        }
        Ok(Self {
            block_size,
            num_disparities,
            min_disparity: 0,
        })
    }

    /// Sets the minimum disparity value (default 0).
    pub fn with_min_disparity(mut self, min_disparity: i32) -> Self {
        self.min_disparity = min_disparity;
        self
    }

    /// Computes a disparity map from a stereo pair of grayscale images.
    ///
    /// Both images must be single-channel (grayscale) `Image`s of identical dimensions.
    /// Returns a 2D tensor of shape `[H, W]` containing disparity values scaled by 16
    /// (fixed-point Q4 format, matching OpenCV convention).
    pub fn compute<B: Backend>(
        &self,
        left: &Image<B>,
        right: &Image<B>,
    ) -> Result<Tensor<B, 2>> {
        if left.channels() != 1 || right.channels() != 1 {
            return Err(IrisError::InvalidParameter(
                "Stereo input images must be single-channel (grayscale)".into(),
            ));
        }
        if left.shape() != right.shape() {
            return Err(IrisError::DimensionMismatch {
                expected: left.shape().to_vec(),
                actual: right.shape().to_vec(),
            });
        }

        let dims = left.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let half_block = self.block_size / 2;
        let max_disp = self.num_disparities;
        let min_d = self.min_disparity;

        let left_data: Vec<f32> = left.tensor.clone().into_data().iter::<f32>().collect();
        let right_data: Vec<f32> = right.tensor.clone().into_data().iter::<f32>().collect();

        let mut disparity = vec![0.0f32; h * w];

        for y in 0..h {
            for x in 0..w {
                let mut best_disp = 0i32;
                let mut best_cost = f32::MAX;

                for d in min_d..(min_d + max_disp) {
                    // SAD over the block
                    let mut cost = 0.0f32;
                    let mut valid = true;

                    for by in -half_block..=half_block {
                        for bx in -half_block..=half_block {
                            let ly = y as i32 + by;
                            let lx = x as i32 + bx;
                            let rx = lx - d;

                            if ly < 0 || ly >= h as i32 || lx < 0 || lx >= w as i32 {
                                valid = false;
                                break;
                            }
                            if rx < 0 || rx >= w as i32 {
                                valid = false;
                                break;
                            }

                            let l_val = left_data[ly as usize * w + lx as usize];
                            let r_val = right_data[ly as usize * w + rx as usize];
                            cost += (l_val - r_val).abs();
                        }
                        if !valid {
                            break;
                        }
                    }

                    if valid && cost < best_cost {
                        best_cost = cost;
                        best_disp = d;
                    }
                }

                // Scale by 16 for fixed-point Q4 output (OpenCV convention)
                disparity[y * w + x] = (best_disp as f32) * 16.0;
            }
        }

        let device = left.tensor.device();
        let data = TensorData::new(disparity, [h, w]);
        let tensor = Tensor::<B, 2>::from_data(data, &device);
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::TensorData;

    #[test]
    fn test_stereo_new_valid() {
        let matcher = StereoBlockMatcher::new(7, 64).unwrap();
        assert_eq!(matcher.block_size, 7);
        assert_eq!(matcher.num_disparities, 64);
        assert_eq!(matcher.min_disparity, 0);
    }

    #[test]
    fn test_stereo_new_invalid_block_size() {
        assert!(StereoBlockMatcher::new(4, 64).is_err());
        assert!(StereoBlockMatcher::new(2, 64).is_err());
    }

    #[test]
    fn test_stereo_new_invalid_disparities() {
        assert!(StereoBlockMatcher::new(7, 0).is_err());
        assert!(StereoBlockMatcher::new(7, 7).is_err());
    }

    #[test]
    fn test_stereo_with_min_disparity() {
        let matcher = StereoBlockMatcher::new(3, 32).unwrap().with_min_disparity(5);
        assert_eq!(matcher.min_disparity, 5);
    }

    #[test]
    fn test_stereo_compute_uniform() {
        let device = test_device();
        let w = 32;
        let h = 16;
        // Both images are identical uniform => disparity should be 0 everywhere
        let flat = vec![0.5f32; h * w];
        let left = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(flat.clone(), [1, h, w]),
            &device,
        ));
        let right = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(flat, [1, h, w]),
            &device,
        ));

        let matcher = StereoBlockMatcher::new(3, 16).unwrap();
        let disp = matcher.compute(&left, &right).unwrap();

        assert_eq!(disp.dims(), [h, w]);
        let vals: Vec<f32> = disp.into_data().iter::<f32>().collect();
        assert!(vals.iter().all(|&v| v.abs() < 1e-5));
    }

    #[test]
    fn test_stereo_compute_shifted() {
        let device = test_device();
        let w = 48;
        let h = 16;
        let mut left_vals = vec![0.0f32; h * w];
        let mut right_vals = vec![0.0f32; h * w];

        // Place a vertical bar in left image at x=24
        for y in 0..h {
            left_vals[y * w + 24] = 1.0;
        }
        // Place same bar in right image at x=20 (shift of 4 pixels)
        for y in 0..h {
            right_vals[y * w + 20] = 1.0;
        }

        let left = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(left_vals, [1, h, w]),
            &device,
        ));
        let right = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(right_vals, [1, h, w]),
            &device,
        ));

        let matcher = StereoBlockMatcher::new(3, 32).unwrap();
        let disp = matcher.compute(&left, &right).unwrap();
        let vals: Vec<f32> = disp.into_data().iter::<f32>().collect();

        // At the bar location, disparity should be 4*16 = 64.0
        let center_disp = vals[(h / 2) * w + 24];
        assert!(
            (center_disp - 64.0).abs() < 1.0,
            "Expected ~64.0, got {center_disp}"
        );
    }

    #[test]
    fn test_stereo_shape_mismatch() {
        let device = test_device();
        let left = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.5f32; 8 * 8], [1, 8, 8]),
            &device,
        ));
        let right = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.5f32; 10 * 10], [1, 10, 10]),
            &device,
        ));

        let matcher = StereoBlockMatcher::new(3, 16).unwrap();
        assert!(matcher.compute(&left, &right).is_err());
    }
}
