use crate::error::{IrisError, Result};
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

impl<B: Backend> Image<B> {
    /// Adds Gaussian noise with the given mean and standard deviation.
    pub fn add_gaussian_noise(&self, mean: f32, std_dev: f32) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        // Box-Muller transform for Gaussian random numbers
        let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next_gaussian: Option<f32> = None;

        for i in 0..(c * h * w) {
            let gaussian = if let Some(g) = next_gaussian.take() {
                g
            } else {
                // Marsaglia polar method
                loop {
                    let u1 = {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
                    };
                    let u2 = {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
                    };
                    let s = u1 * u1 + u2 * u2;
                    if s > 0.0 && s < 1.0 {
                        let factor = (-2.0 * s.ln() / s).sqrt();
                        let g1 = u1 * factor;
                        let g2 = u2 * factor;
                        next_gaussian = Some(g2);
                        break g1;
                    }
                }
            };

            let noise = mean + std_dev * gaussian;
            out_vals[i] = (flat_vals[i] + noise).clamp(0.0, 1.0);
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Adds salt-and-pepper (impulse) noise with the given probability.
    /// `amount` is the fraction of pixels to corrupt (0.0 to 1.0).
    pub fn add_salt_pepper_noise(&self, amount: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&amount) {
            return Err(IrisError::InvalidParameter(
                "amount must be in [0.0, 1.0]".into(),
            ));
        }

        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = flat_vals.clone();

        let total_pixels = h * w;
        let num_noise = (total_pixels as f32 * amount) as usize;

        let mut seed: u64 = 0xABCD_EF01_2345_6789;

        for _ in 0..num_noise {
            let py = {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as usize) % h
            };
            let px = {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as usize) % w
            };
            let is_salt = {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 33) & 1 == 0
            };

            for ch in 0..c {
                out_vals[ch * total_pixels + py * w + px] = if is_salt { 1.0 } else { 0.0 };
            }
        }

        let new_data = TensorData::new(out_vals, [c, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }

    /// Adds speckle (multiplicative) noise: pixel = pixel + pixel * noise.
    pub fn add_speckle_noise(&self, std_dev: f32) -> Result<Self> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let device = self.tensor.device();
        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; c * h * w];

        let mut seed: u64 = 0x1111_2222_3333_4444;
        let mut next_gaussian: Option<f32> = None;

        for i in 0..(c * h * w) {
            let gaussian = if let Some(g) = next_gaussian.take() {
                g
            } else {
                loop {
                    let u1 = {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
                    };
                    let u2 = {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
                    };
                    let s = u1 * u1 + u2 * u2;
                    if s > 0.0 && s < 1.0 {
                        let factor = (-2.0 * s.ln() / s).sqrt();
                        let g1 = u1 * factor;
                        let g2 = u2 * factor;
                        next_gaussian = Some(g2);
                        break g1;
                    }
                }
            };

            let noise = flat_vals[i] * std_dev * gaussian;
            out_vals[i] = (flat_vals[i] + noise).clamp(0.0, 1.0);
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

    #[test]
    fn test_gaussian_noise() {
        let device = test_device();
        let data = TensorData::new(vec![0.5f32; 3 * 8 * 8], [3, 8, 8]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(data, &device));
        let noisy = img.add_gaussian_noise(0.0, 0.05).unwrap();
        assert_eq!(noisy.shape(), [3, 8, 8]);
    }

    #[test]
    fn test_salt_pepper_noise() {
        let device = test_device();
        let data = TensorData::new(vec![0.5f32; 3 * 8 * 8], [3, 8, 8]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(data, &device));
        let noisy = img.add_salt_pepper_noise(0.1).unwrap();
        assert_eq!(noisy.shape(), [3, 8, 8]);
    }

    #[test]
    fn test_speckle_noise() {
        let device = test_device();
        let data = TensorData::new(vec![0.5f32; 3 * 8 * 8], [3, 8, 8]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(data, &device));
        let noisy = img.add_speckle_noise(0.1).unwrap();
        assert_eq!(noisy.shape(), [3, 8, 8]);
    }

    #[test]
    fn test_noise_invalid_amount() {
        let device = test_device();
        let data = TensorData::new(vec![0.5f32; 3 * 8 * 8], [3, 8, 8]);
        let img = Image::new(Tensor::<TestBackend, 3>::from_data(data, &device));
        assert!(img.add_salt_pepper_noise(1.5).is_err());
    }
}
