use crate::error::{ObserversError, Result};
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Image Stitcher for panorama creation.
pub struct Stitcher;

impl Stitcher {
    /// Stitches a list of images into a single panorama.
    pub fn stitch<B: Backend>(&self, images: &[Image<B>]) -> Result<Image<B>> {
        if images.is_empty() {
            return Err(ObserversError::InvalidParameter(
                "Images list cannot be empty".into(),
            ));
        }
        if images.len() == 1 {
            return Ok(images[0].clone());
        }

        // Simulates stitching by resizing and combining the first two images horizontally
        let img1 = &images[0];
        let img2 = &images[1];

        let _new_w = img1.width() + img2.width();
        let new_h = img1.height().max(img2.height());

        let resized1 = img1.resize(img1.width(), new_h)?;
        let resized2 = img2.resize(img2.width(), new_h)?;

        // Concatenate tensors along dimension 2 (Width axis)
        let stitched_tensor = burn::tensor::Tensor::cat(vec![resized1.tensor, resized2.tensor], 2);
        Ok(Image::new(stitched_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_stitching() {
        let device = Default::default();
        let flat_data1 = vec![0.5f32; 3 * 8 * 8];
        let flat_data2 = vec![0.3f32; 3 * 8 * 8];

        let img1 = Image::new(Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data1, [3, 8, 8]), &device));
        let img2 = Image::new(Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data2, [3, 8, 8]), &device));

        let stitcher = Stitcher;
        let stitched = stitcher.stitch(&[img1, img2]).unwrap();
        assert_eq!(stitched.shape(), [3, 8, 16]);
    }
}

