use crate::error::Result;
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Background subtractor pipeline.
pub struct BackgroundSubtractor<B: Backend> {
    pub learning_rate: f32,
    pub threshold: f32,
    background: Option<Image<B>>,
}

impl<B: Backend> BackgroundSubtractor<B> {
    /// Creates a new background subtractor.
    #[must_use]
    pub fn new(learning_rate: f32, threshold: f32) -> Self {
        Self {
            learning_rate,
            threshold,
            background: None,
        }
    }

    /// Processes a new frame, returning the foreground binary mask.
    pub fn apply(&mut self, frame: &Image<B>) -> Result<Image<B>> {
        let frame_gray = frame.grayscale()?;

        let bg = if let Some(bg_img) = &self.background {
            // Update running background model: B_t = (1 - alpha)*B_{t-1} + alpha*I_t
            let updated = bg_img
                .tensor
                .clone()
                .mul_scalar(1.0 - self.learning_rate)
                .add(frame_gray.tensor.clone().mul_scalar(self.learning_rate));
            let bg_new = Image::new(updated);
            self.background = Some(bg_new.clone());
            bg_new
        } else {
            self.background = Some(frame_gray.clone());
            frame_gray.clone()
        };

        // Foreground mask: F = |Frame - Background| > threshold
        let diff = frame_gray.absdiff(&bg)?;
        let mask = diff.threshold(self.threshold, 1.0, crate::threshold::ThresholdType::Binary)?;
        Ok(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_background_subtractor() {
        let device = test_device();
        let flat_data1 = vec![0.5f32; 3 * 8 * 8];
        let flat_data2 = vec![0.6f32; 3 * 8 * 8];

        let img1 = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(flat_data1, [3, 8, 8]),
            &device,
        ));
        let img2 = Image::new(Tensor::<TestBackend, 3>::from_data(
            TensorData::new(flat_data2, [3, 8, 8]),
            &device,
        ));

        let mut bs = BackgroundSubtractor::new(0.1, 0.05);
        let mask1 = bs.apply(&img1).unwrap();
        assert_eq!(mask1.shape(), [1, 8, 8]);

        let mask2 = bs.apply(&img2).unwrap();
        assert_eq!(mask2.shape(), [1, 8, 8]);
    }
}
