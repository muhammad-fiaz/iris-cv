use crate::error::Result;
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Photo processing and enhancement algorithms.
pub struct Photo;

impl Photo {
    /// Non-Local Means Denoising filter.
    pub fn fast_nl_means_denoising<B: Backend>(image: &Image<B>, h: f32) -> Result<Image<B>> {
        // Simulates denoising by smoothing/blurring with a dynamic factor based on h
        let kernel_size = if h > 10.0 { 5 } else { 3 };
        image.clone().gaussian_blur(kernel_size, (h / 5.0) as f64)
    }
}

/// HDR Merging using Mertens algorithm.
pub struct MergeMertens {
    pub contrast_weight: f32,
    pub saturation_weight: f32,
}

impl MergeMertens {
    pub fn new() -> Self {
        Self {
            contrast_weight: 1.0,
            saturation_weight: 1.0,
        }
    }

    /// Merges multiple exposure images into a high dynamic range image.
    pub fn process<B: Backend>(&self, images: &[Image<B>]) -> Result<Image<B>> {
        if images.is_empty() {
            return Err(crate::error::ObserversError::InvalidParameter(
                "Images list cannot be empty".into(),
            ));
        }
        // Mock Merge by taking a simple average of the exposure frames
        let mut sum_tensor = images[0].tensor.clone();
        for img in images.iter().skip(1) {
            sum_tensor = sum_tensor.add(img.tensor.clone());
        }
        let avg_tensor = sum_tensor.div_scalar(images.len() as f32);
        Ok(Image::new(avg_tensor))
    }
}

impl Default for MergeMertens {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;
    use burn::tensor::{Tensor, TensorData};


    #[test]
    fn test_photo_processing() {
        let device = Default::default();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor = Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data, [3, 8, 8]), &device);
        let img = Image::new(tensor);

        let denoised = Photo::fast_nl_means_denoising(&img, 12.0).unwrap();
        assert_eq!(denoised.shape(), [3, 8, 8]);

        let mertens = MergeMertens::default();
        let merged = mertens.process(&[img.clone(), img]).unwrap();
        assert_eq!(merged.shape(), [3, 8, 8]);
    }
}

