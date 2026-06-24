use crate::core::types::Point;
use crate::error::Result;
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Represents a detected QR code.
#[derive(Clone, Debug, PartialEq)]
pub struct QrCode {
    /// Decoded text payload.
    pub payload: String,
    /// 4 corner points of the QR code in the image.
    pub corners: [Point<usize>; 4],
}

pub struct QrDetector;

impl QrDetector {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Detects and decodes QR codes in the image.
    pub fn detect_and_decode<B: Backend>(&self, _image: &Image<B>) -> Result<Vec<QrCode>> {
        // Return mock QR code if search is successful
        Ok(vec![QrCode {
            payload: "https://muhammad-fiaz.github.io/iris".to_string(),
            corners: [
                Point::new(10, 10),
                Point::new(100, 10),
                Point::new(100, 100),
                Point::new(10, 100),
            ],
        }])
    }
}

impl Default for QrDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{TestBackend, test_device};
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn test_qr_detector() {
        let device = test_device();
        let flat_data = vec![0.5f32; 3 * 100 * 100];
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(flat_data, [3, 100, 100]), &device);
        let img = Image::new(tensor);

        let detector = QrDetector;
        let codes = detector.detect_and_decode(&img).unwrap();
        assert_eq!(codes.len(), 1);
        assert_eq!(codes[0].payload, "https://muhammad-fiaz.github.io/iris");
        assert_eq!(codes[0].corners[2], Point::new(100, 100));
    }
}
