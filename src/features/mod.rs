pub mod matching;

pub use matching::{BFMatcher, DMatch, MatchDrawer};

use crate::core::types::Point;
use crate::error::Result;
use crate::image::Image;
use burn::tensor::{Tensor, backend::Backend};

/// Represents a detected keypoint in an image.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyPoint {
    /// Coordinates of the keypoint.
    pub pt: Point<f64>,
    /// Diameter of the meaningful keypoint neighborhood.
    pub size: f64,
    /// Computed orientation of the keypoint (degrees).
    pub angle: f64,
    /// Strength/response of the keypoint.
    pub response: f64,
    /// Octave (pyramid layer) from which the keypoint was extracted.
    pub octave: i32,
    /// Object class ID.
    pub class_id: i32,
}

impl KeyPoint {
    pub fn new(x: f64, y: f64, size: f64) -> Self {
        Self {
            pt: Point::new(x, y),
            size,
            angle: -1.0,
            response: 0.0,
            octave: 0,
            class_id: -1,
        }
    }
}

/// Feature detector type.
pub enum FeatureType {
    ORB,
    BRISK,
    AKAZE,
    SIFT,
}

pub struct FeatureDetector {
    detector_type: FeatureType,
}

impl FeatureDetector {
    pub fn new(detector_type: FeatureType) -> Self {
        Self { detector_type }
    }

    /// Detects keypoints in an image.
    pub fn detect<B: Backend>(&self, image: &Image<B>) -> Result<Vec<KeyPoint>> {
        // Return dummy keypoints for API structure demonstration
        let w = image.width() as f64;
        let h = image.height() as f64;
        let kp1 = KeyPoint::new(w * 0.25, h * 0.25, 10.0);
        let kp2 = KeyPoint::new(w * 0.75, h * 0.75, 15.0);
        Ok(vec![kp1, kp2])
    }

    /// Computes descriptors for detected keypoints.
    /// Returns a descriptor tensor of shape [NumKeyPoints, DescriptorDim].
    pub fn compute<B: Backend>(
        &self,
        image: &Image<B>,
        keypoints: &[KeyPoint],
    ) -> Result<Tensor<B, 2>> {
        let n = keypoints.len();
        let dim = match self.detector_type {
            FeatureType::ORB => 32,
            FeatureType::BRISK => 64,
            FeatureType::AKAZE => 61,
            FeatureType::SIFT => 128,
        };
        let device = image.tensor.device();
        Ok(Tensor::<B, 2>::zeros([n, dim], &device))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;
    use burn::tensor::TensorData;

    #[test]
    fn test_feature_detector() {
        let device = Default::default();
        let flat_data = vec![0.5f32; 3 * 100 * 100];
        let tensor = Tensor::<Wgpu, 3>::from_data(TensorData::new(flat_data, [3, 100, 100]), &device);
        let img = Image::new(tensor);

        let detector = FeatureDetector::new(FeatureType::ORB);
        let keypoints = detector.detect(&img).unwrap();
        assert_eq!(keypoints.len(), 2);

        let descriptors = detector.compute(&img, &keypoints).unwrap();
        assert_eq!(descriptors.dims(), [2, 32]);
    }
}

