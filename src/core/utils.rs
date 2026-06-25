use crate::core::types::Point;
use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

impl<B: Backend> Image<B> {
    /// Computes the element-wise sum of two images.
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(IrisError::DimensionMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        let added = self.tensor.clone().add(other.tensor.clone());
        Ok(Image::new(added))
    }

    /// Computes the element-wise difference of two images.
    pub fn subtract(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(IrisError::DimensionMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        let subbed = self.tensor.clone().sub(other.tensor.clone());
        Ok(Image::new(subbed))
    }

    /// Computes the element-wise multiplication of two images.
    pub fn multiply(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(IrisError::DimensionMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        let muled = self.tensor.clone().mul(other.tensor.clone());
        Ok(Image::new(muled))
    }

    /// Computes the element-wise division of two images.
    pub fn divide(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(IrisError::DimensionMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        let dived = self.tensor.clone().div(other.tensor.clone());
        Ok(Image::new(dived))
    }

    /// Computes the absolute difference between two images.
    pub fn absdiff(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(IrisError::DimensionMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        let diff = self.tensor.clone().sub(other.tensor.clone()).abs();
        Ok(Image::new(diff))
    }

    /// Computes bitwise AND of two images at pixel/byte level (0..255).
    pub fn bitwise_and(&self, other: &Self) -> Result<Self> {
        self.bitwise_op(other, |a, b| a & b)
    }

    /// Computes bitwise OR of two images.
    pub fn bitwise_or(&self, other: &Self) -> Result<Self> {
        self.bitwise_op(other, |a, b| a | b)
    }

    /// Computes bitwise XOR of two images.
    pub fn bitwise_xor(&self, other: &Self) -> Result<Self> {
        self.bitwise_op(other, |a, b| a ^ b)
    }

    /// Computes bitwise NOT of the image.
    pub fn bitwise_not(&self) -> Result<Self> {
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
            out_vals.par_iter_mut().enumerate().for_each(|(i, val)| {
                let pixel_val = (flat_vals[i].clamp(0.0, 1.0) * 255.0) as u8;
                *val = f32::from(!pixel_val) / 255.0;
            });
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Returns the average value of all elements in the image.
    pub fn mean(&self) -> Result<Vec<f64>> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut channel_means = vec![0.0; c];

        for ch in 0..c {
            let mut sum = 0.0;
            for y in 0..h {
                for x in 0..w {
                    sum += f64::from(flat_vals[ch * h * w + y * w + x]);
                }
            }
            channel_means[ch] = sum / ((h * w) as f64);
        }

        Ok(channel_means)
    }

    /// Computes the mean and standard deviation of image elements channel-wise.
    pub fn mean_std_dev(&self) -> Result<(Vec<f64>, Vec<f64>)> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut channel_means = vec![0.0; c];
        let mut channel_stddevs = vec![0.0; c];

        let n = (h * w) as f64;

        for ch in 0..c {
            let mut sum = 0.0;
            for y in 0..h {
                for x in 0..w {
                    sum += f64::from(flat_vals[ch * h * w + y * w + x]);
                }
            }
            let mean = sum / n;
            channel_means[ch] = mean;

            let mut sq_sum = 0.0;
            for y in 0..h {
                for x in 0..w {
                    let diff = f64::from(flat_vals[ch * h * w + y * w + x]) - mean;
                    sq_sum += diff * diff;
                }
            }
            channel_stddevs[ch] = (sq_sum / n).sqrt();
        }

        Ok((channel_means, channel_stddevs))
    }

    /// Finds global minimum and maximum values and their coordinate locations in a single-channel image.
    pub fn min_max_loc(&self) -> Result<(f64, f64, Point<usize>, Point<usize>)> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        if c != 1 {
            return Err(IrisError::InvalidParameter(
                "min_max_loc requires a single-channel image".into(),
            ));
        }

        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();

        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;
        let mut min_loc = Point::new(0, 0);
        let mut max_loc = Point::new(0, 0);

        for y in 0..h {
            for x in 0..w {
                let val = f64::from(flat_vals[y * w + x]);
                if val < min_val {
                    min_val = val;
                    min_loc = Point::new(x, y);
                }
                if val > max_val {
                    max_val = val;
                    max_loc = Point::new(x, y);
                }
            }
        }

        Ok((min_val, max_val, min_loc, max_loc))
    }

    /// Counts non-zero pixels (value > 0.0).
    pub fn count_non_zero(&self) -> Result<usize> {
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let count = flat_vals.iter().filter(|&&x| x > 0.0).count();
        Ok(count)
    }

    /// Produces a binary mask where pixels within the range [low, high] are set to 1.0.
    /// For multi-channel images, all channels must be in range for the pixel to be marked.
    pub fn in_range(&self, low: &[f32], high: &[f32]) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        if low.len() != c || high.len() != c {
            return Err(IrisError::InvalidParameter(format!(
                "low/high length ({}/{}) must match channels ({c})",
                low.len(),
                high.len(),
            )));
        }

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; h * w];

        let pixels = h * w;
        for i in 0..pixels {
            let mut in_range = true;
            for ch in 0..c {
                let val = flat_vals[ch * pixels + i];
                if val < low[ch] || val > high[ch] {
                    in_range = false;
                    break;
                }
            }
            out_vals[i] = if in_range { 1.0 } else { 0.0 };
        }

        let new_data = TensorData::new(out_vals, [1, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Normalizes the image to the given range [min_val, max_val].
    pub fn normalize(&self, min_val: f32, max_val: f32) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        for ch in 0..c {
            let mut ch_min = f32::MAX;
            let mut ch_max = f32::MIN;
            let pixels = h * w;
            for i in 0..pixels {
                let v = flat_vals[ch * pixels + i];
                if v < ch_min { ch_min = v; }
                if v > ch_max { ch_max = v; }
            }
            let range = ch_max - ch_min;
            for i in 0..pixels {
                let v = flat_vals[ch * pixels + i];
                out_vals[ch * pixels + i] = if range.abs() < 1e-10 {
                    min_val
                } else {
                    min_val + (v - ch_min) / range * (max_val - min_val)
                };
            }
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    // Helper for pixel-wise logical operations
    fn bitwise_op(&self, other: &Self, op: impl Fn(u8, u8) -> u8 + Sync + Send) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(IrisError::DimensionMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let data_self = self.tensor.clone().into_data();
        let data_other = other.tensor.clone().into_data();

        let vals_self: Vec<f32> = data_self.iter::<f32>().collect();
        let vals_other: Vec<f32> = data_other.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        {
            use rayon::prelude::*;
            out_vals.par_iter_mut().enumerate().for_each(|(i, val)| {
                let b1 = (vals_self[i].clamp(0.0, 1.0) * 255.0) as u8;
                let b2 = (vals_other[i].clamp(0.0, 1.0) * 255.0) as u8;
                *val = f32::from(op(b1, b2)) / 255.0;
            });
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::backend::ndarray::NdArrayDevice;

    fn get_test_device() -> NdArrayDevice {
        test_device()
    }

    #[test]
    fn test_image_math_and_bitwise() {
        let device = get_test_device();
        let data1 = TensorData::new(vec![0.5f32; 3 * 4 * 4], [3, 4, 4]);
        let data2 = TensorData::new(vec![0.2f32; 3 * 4 * 4], [3, 4, 4]);

        let img1 = Image::new(Tensor::<TestBackend, 3>::from_data(data1, &device));
        let img2 = Image::new(Tensor::<TestBackend, 3>::from_data(data2, &device));

        // Math
        let added = img1.add(&img2).unwrap();
        assert_eq!(added.shape(), [3, 4, 4]);

        let subbed = img1.subtract(&img2).unwrap();
        assert_eq!(subbed.shape(), [3, 4, 4]);

        let absdiff = img1.absdiff(&img2).unwrap();
        assert_eq!(absdiff.shape(), [3, 4, 4]);

        // Bitwise
        let bit_and = img1.bitwise_and(&img2).unwrap();
        assert_eq!(bit_and.shape(), [3, 4, 4]);

        let bit_not = img1.bitwise_not().unwrap();
        assert_eq!(bit_not.shape(), [3, 4, 4]);

        // Stats
        let mean_vals = img1.mean().unwrap();
        assert!((mean_vals[0] - 0.5).abs() < 1e-4);

        let count = img1.count_non_zero().unwrap();
        assert_eq!(count, 3 * 4 * 4);
    }

    #[test]
    fn test_in_range() {
        let device = get_test_device();
        let data = vec![0.1, 0.5, 0.9, 0.3, 0.6, 0.2, 0.7, 0.8, 0.4];
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(data, [3, 1, 3]),
            &device,
        ));

        let mask = img.in_range(&[0.2, 0.2, 0.2], &[0.8, 0.8, 0.8]).unwrap();
        assert_eq!(mask.shape(), [1, 1, 3]);
        let vals: Vec<f32> = mask.tensor.into_data().iter::<f32>().collect();
        // pixel 0: 0.1<0.2 -> 0.0; pixel 1: 0.5 in range -> 1.0; pixel 2: 0.9>0.8 -> 0.0
        assert!((vals[0]).abs() < 1e-5);
        assert!((vals[1] - 1.0).abs() < 1e-5);
        assert!((vals[2]).abs() < 1e-5);
    }

    #[test]
    fn test_normalize() {
        let device = get_test_device();
        let data = vec![0.2, 0.4, 0.6, 0.8];
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(data, [1, 1, 4]),
            &device,
        ));

        let normalized = img.normalize(0.0, 1.0).unwrap();
        assert_eq!(normalized.shape(), [1, 1, 4]);
        let vals: Vec<f32> = normalized.tensor.into_data().iter::<f32>().collect();
        assert!((vals[0]).abs() < 1e-5); // min => 0.0
        assert!((vals[3] - 1.0).abs() < 1e-5); // max => 1.0
    }

    #[test]
    fn test_in_range_invalid_length() {
        let device = get_test_device();
        let data = vec![0.5f32; 3 * 4 * 4];
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(data, [3, 4, 4]),
            &device,
        ));
        assert!(img.in_range(&[0.0], &[0.5, 0.5, 0.5]).is_err());
    }
}
