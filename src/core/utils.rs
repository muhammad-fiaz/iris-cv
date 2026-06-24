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

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out_vals.par_iter_mut().enumerate().for_each(|(i, val)| {
                let pixel_val = (flat_vals[i].clamp(0.0, 1.0) * 255.0) as u8;
                *val = f32::from(!pixel_val) / 255.0;
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..(c * h * w) {
                let pixel_val = (flat_vals[i].clamp(0.0, 1.0) * 255.0) as u8;
                out_vals[i] = (!pixel_val) as f32 / 255.0;
            }
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

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out_vals.par_iter_mut().enumerate().for_each(|(i, val)| {
                let b1 = (vals_self[i].clamp(0.0, 1.0) * 255.0) as u8;
                let b2 = (vals_other[i].clamp(0.0, 1.0) * 255.0) as u8;
                *val = f32::from(op(b1, b2)) / 255.0;
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..(c * h * w) {
                let b1 = (vals_self[i].clamp(0.0, 1.0) * 255.0) as u8;
                let b2 = (vals_other[i].clamp(0.0, 1.0) * 255.0) as u8;
                out_vals[i] = op(b1, b2) as f32 / 255.0;
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
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    type TestBackend = Wgpu;

    fn get_test_device() -> WgpuDevice {
        Default::default()
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
}
