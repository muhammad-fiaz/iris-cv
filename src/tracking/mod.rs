pub mod subtractor;

pub use subtractor::BackgroundSubtractor;

use crate::core::types::Rect;
use crate::error::Result;
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Standard tracker algorithms.
pub enum TrackerType {
    KCF,
    CSRT,
    MOSSE,
}

/// Object tracker pipeline.
pub struct Tracker<B: Backend> {
    pub tracker_type: TrackerType,
    pub bbox: Option<Rect<usize>>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Tracker<B> {
    /// Creates a new Tracker.
    #[must_use]
    pub fn new(tracker_type: TrackerType) -> Self {
        Self {
            tracker_type,
            bbox: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Initializes the tracker with a known bounding box of the target object.
    pub fn init(&mut self, _image: &Image<B>, bbox: Rect<usize>) -> Result<()> {
        self.bbox = Some(bbox);
        Ok(())
    }

    /// Updates the tracker, finding the new location of the target in the frame.
    pub fn update(&mut self, _image: &Image<B>) -> Result<Rect<usize>> {
        let current = self.bbox.ok_or_else(|| {
            crate::error::IrisError::Generic("Tracker not initialized. Call init first.".into())
        })?;

        // Simulates a tiny random motion/update for demonstration
        let updated = Rect::new(current.x + 1, current.y + 1, current.width, current.height);
        self.bbox = Some(updated);
        Ok(updated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_object_tracker() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 8 * 8];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 8, 8]), &device);
        let img = Image::new(tensor);

        let mut tracker = Tracker::new(TrackerType::KCF);
        let init_bbox = Rect::new(2, 2, 4, 4);
        tracker.init(&img, init_bbox).unwrap();

        let updated = tracker.update(&img).unwrap();
        assert_eq!(updated, Rect::new(3, 3, 4, 4));
    }
}
