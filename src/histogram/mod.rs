use crate::error::Result;
use crate::image::Image;
use burn::tensor::{Tensor, TensorData, backend::Backend};

impl<B: Backend> Image<B> {
    /// Computes the 256-bin histogram for each channel.
    /// Returns a vector of vectors (one per channel), each containing 256 counts.
    pub fn calc_hist(&self) -> Result<Vec<Vec<u32>>> {
        let dims = self.tensor.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];

        let tensor_data = self.tensor.clone().into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut histograms = vec![vec![0u32; 256]; c];

        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let val = flat_vals[ch * h * w + y * w + x];
                    let bin = (val.clamp(0.0, 1.0) * 255.0) as usize;
                    histograms[ch][bin] += 1;
                }
            }
        }

        Ok(histograms)
    }

    /// Performs histogram equalization on a grayscale image to enhance contrast.
    pub fn equalize_hist(&self) -> Result<Self> {
        let gray = self.grayscale()?;
        let dims = gray.tensor.dims();
        let h = dims[1];
        let w = dims[2];

        let device = gray.tensor.device();
        let tensor_data = gray.tensor.into_data();
        let flat_vals: Vec<f32> = tensor_data.iter::<f32>().collect();
        let mut out_vals = vec![0.0f32; h * w];

        // 1. Compute histogram
        let mut hist = [0u32; 256];
        for &val in &flat_vals {
            let bin = (val.clamp(0.0, 1.0) * 255.0) as usize;
            hist[bin] += 1;
        }

        // 2. Compute cumulative distribution function (CDF)
        let mut cdf = [0u32; 256];
        let mut sum = 0u32;
        for i in 0..256 {
            sum += hist[i];
            cdf[i] = sum;
        }

        // 3. Find CDF minimum non-zero value
        let cdf_min = cdf.iter().find(|&&x| x > 0).copied().unwrap_or(0) as f32;
        let total = (h * w) as f32;

        // 4. Equalize mapping
        let mut lut = [0.0f32; 256];
        if total > cdf_min {
            for i in 0..256 {
                lut[i] = ((cdf[i] as f32 - cdf_min) / (total - cdf_min) * 255.0).round() / 255.0;
            }
        }

        for i in 0..(h * w) {
            let bin = (flat_vals[i].clamp(0.0, 1.0) * 255.0) as usize;
            out_vals[i] = lut[bin];
        }

        let new_data = TensorData::new(out_vals, [1, h, w]);
        let new_tensor = Tensor::<B, 3>::from_data(new_data, &device);
        Ok(Image::new(new_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;

    #[test]
    fn test_histogram_operations() {
        let device = Default::default();
        let flat_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let tensor = Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data, [1, 2, 4]), &device);
        let img = Image::new(tensor);

        let hists = img.calc_hist().unwrap();
        assert_eq!(hists.len(), 1);
        assert_eq!(hists[0].len(), 256);

        let eq = img.equalize_hist().unwrap();
        assert_eq!(eq.shape(), [1, 2, 4]);
    }
}

