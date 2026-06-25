use crate::image::Image;
use burn::tensor::backend::Backend;
use std::time::Duration;

/// A single video frame with timing metadata.
///
/// Wraps an `Image` with presentation timestamp and duration information
/// for proper video playback and editing workflows.
#[derive(Clone, Debug)]
pub struct Frame<B: Backend> {
    /// The image data for this frame (C, H, W tensor).
    pub image: Image<B>,
    /// Presentation timestamp — when this frame should be displayed.
    pub pts: Duration,
    /// How long this frame should be displayed (used for variable frame rate).
    pub duration: Duration,
    /// Frame index in the sequence (0-based).
    pub index: usize,
    /// Whether this frame is a keyframe (I-frame).
    pub is_keyframe: bool,
}

impl<B: Backend> Frame<B> {
    /// Creates a new frame with the given image and timestamp.
    #[must_use]
    pub fn new(image: Image<B>, pts: Duration, index: usize) -> Self {
        Self {
            image,
            pts,
            duration: Duration::ZERO,
            index,
            is_keyframe: false,
        }
    }

    /// Creates a keyframe (I-frame) at the given timestamp.
    #[must_use]
    pub fn keyframe(image: Image<B>, pts: Duration, index: usize) -> Self {
        Self {
            image,
            pts,
            duration: Duration::ZERO,
            index,
            is_keyframe: true,
        }
    }

    /// Sets the display duration for this frame (for variable frame rate).
    #[must_use]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Returns the width of the frame in pixels.
    #[must_use]
    pub fn width(&self) -> usize {
        self.image.width()
    }

    /// Returns the height of the frame in pixels.
    #[must_use]
    pub fn height(&self) -> usize {
        self.image.height()
    }

    /// Returns the number of color channels.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.image.channels()
    }

    /// Returns the frame shape as [C, H, W].
    #[must_use]
    pub fn shape(&self) -> [usize; 3] {
        self.image.shape()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_frame_creation() {
        let device = test_device();
        let data = TensorData::new(vec![0.5f32; 3 * 64 * 64], [3, 64, 64]);
        let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);
        let img = Image::new(tensor);

        let frame = Frame::new(img, Duration::from_millis(33), 0);
        assert_eq!(frame.width(), 64);
        assert_eq!(frame.height(), 64);
        assert_eq!(frame.channels(), 3);
        assert_eq!(frame.index, 0);
        assert!(!frame.is_keyframe);

        let kf = Frame::keyframe(frame.image.clone(), Duration::from_millis(0), 0);
        assert!(kf.is_keyframe);
    }

    #[test]
    fn test_frame_with_duration() {
        let device = test_device();
        let data = TensorData::new(vec![0.5f32; 3 * 32 * 32], [3, 32, 32]);
        let tensor = Tensor::<TestBackend, 3>::from_data(data, &device);
        let img = Image::new(tensor);

        let frame =
            Frame::new(img, Duration::from_millis(0), 0).with_duration(Duration::from_millis(33));
        assert_eq!(frame.duration, Duration::from_millis(33));
    }
}
